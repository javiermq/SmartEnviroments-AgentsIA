"""Historias aisladas por usuario y sesión, compartiendo el mismo Ollama."""
from pathlib import Path
from threading import RLock, Thread
from collections import deque
from copy import deepcopy
import logging
import time
from .context import DATA_DIR, build_context, parse_timestamp
from .conversation_backend import ConversationBackend


def user_id(value):
    if not isinstance(value, str) or value.lower() not in {'javi', 'mariola'}:
        raise ValueError('Usuario permitido: javi o mariola')
    return value.lower()


class UserSessions:
    def __init__(self, ollama_url, data_root=DATA_DIR):
        self.ollama_url, self.data_root = ollama_url, Path(data_root)
        self.sessions = {}
        self.lock = RLock()

    def _session(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('El cuerpo debe ser un objeto JSON')
        user = user_id(payload.get('user'))
        session = payload.get('session_id', 'reachy')
        if not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError('session_id inválido')
        t0 = payload.get('t0')
        model = payload.get('model', 'qwen3:4b-instruct-2507-q4_K_M')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model inválido')
        with self.lock:
            key = (session, user)
            if key not in self.sessions:
                if not isinstance(t0, str):
                    raise ValueError('Se requiere t0 para iniciar la sesión')
                parse_timestamp(t0)
                self.sessions[key] = LiveSession(ConversationBackend(
                    t0, model, self.ollama_url, self.data_root / user))
            state = self.sessions[key]
            if state.backend.model != model:
                raise ValueError('El modelo de esta sesión ya está fijado')
        return user, state

    def respond(self, payload, trace=None):
        text = payload.get('text') if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Se requiere text no vacío')
        user, state = self._session(payload)
        # Registrar la voz antes de esperar al LLM: invalida comentarios pendientes.
        with state.lock:
            state.turns.append(time.monotonic())
            state.timeline.append({'timestamp': state.context['t0'], 'source': 'user', 'text': text})
            state.revision += 1
            state.outbox = None
            state.voice_pending += 1
        try:
            return self._voice_response(user, state, text, trace)
        finally:
            with state.lock:
                state.voice_pending -= 1

    def _voice_response(self, user, state, text, trace):
        with state.generation_lock:
            with state.lock:
                backend = state.backend
                backend.update_environment(state.context_with_conversation())
                state.trim_history()
            events = []
            def record(event, data):
                events.append({'event': event, 'data': data})
                if trace:
                    trace(event, data)
            answer = backend.respond(text, record)
            with state.lock:
                state.revision += 1
                state.outbox = None
                state.timeline.append({'timestamp': backend.t0, 'source': 'assistant', 'text': answer})
            return {'text': answer, 'gesture': backend.last_gesture, 'user': user,
                    'trace': events, 't0': backend.t0}

    def activate(self, payload):
        user, state = self._session(payload)
        with self.lock:
            for (session, other_user), other in self.sessions.items():
                if session == payload.get('session_id', 'reachy'):
                    with other.lock:
                        other.active = other_user == user
                        other.outbox = None
                        other.revision += 1
        with state.lock:
            state.environment_pending = True
            if not state.worker_running:
                state.worker_running = True
                Thread(target=self._evaluate, args=(user, state), daemon=True).start()
        return {'user': user, 't0': state.context['t0']}

    def deactivate(self, payload):
        user, state = self._session(payload)
        with state.lock:
            state.active = False
            state.revision += 1
            state.outbox = None
        return {'user': user, 'action': 'silent'}

    def environment(self, payload):
        user, state = self._session(payload)
        context = payload.get('context')
        if not isinstance(context, dict) or not isinstance(context.get('t0'), str):
            raise ValueError('Se requiere context con t0')
        moment = parse_timestamp(context['t0'])
        # Perfil y reglas proceden del servidor; el emisor solo actualiza observaciones.
        supplied = {key: deepcopy(context[key]) for key in (
            'sensors_at_t0', 'watch_minute_sample_NOT_TOTALS', 'current_activities',
            'history_start', 'activities_last_36h') if key in context}
        if len(supplied) != 5:
            raise ValueError('El entorno debe incluir sensores y HAR de 36 horas')
        import json
        json.dumps(supplied, allow_nan=False)
        if any(parse_timestamp(item['end_time']) > moment or parse_timestamp(item['start_time']) > moment
               for item in supplied['activities_last_36h']):
            raise ValueError('HAR futuro no permitido')
        if any('end_time' in item or parse_timestamp(item['start_time']) > moment
               for item in supplied['current_activities']):
            raise ValueError('Actividad actual inválida o futura')
        for sample in (supplied['sensors_at_t0'], supplied['watch_minute_sample_NOT_TOTALS']):
            if sample is not None and not 0 <= (moment - parse_timestamp(sample['timestamp'])).total_seconds() < 60:
                raise ValueError('La fila de sensores no corresponde al minuto actual')
        sample = supplied['watch_minute_sample_NOT_TOTALS']
        supplied['environment_row'] = None if sample is None else {
            **supplied['sensors_at_t0'], 'steps': sample['steps_in_this_minute'],
            'distance_m': sample['distance_m_in_this_minute'], 'sleep': sample['sleep_indicator_0_or_1']}
        with state.lock:
            if moment < parse_timestamp(state.context['t0']):
                raise ValueError('No se puede retroceder el reloj de la sesión')
            previous = state.context
            state.context = {**previous, **supplied, 't0': context['t0']}
            state.context['environment_changes'] = {
                key: {'before': previous.get(key), 'now': state.context.get(key)}
                for key in ('sensors_at_t0', 'current_activities')
                if previous.get(key) != state.context.get(key)}
            state.revision += 1
            state.outbox = None
            state.environment_pending = state.active
            if state.active and not state.worker_running:
                state.worker_running = True
                Thread(target=self._evaluate, args=(user, state), daemon=True).start()
        return {'user': user, 't0': context['t0'], 'accepted': True}

    def _evaluate(self, user, state):
        try:
            while True:
                with state.generation_lock:
                    with state.lock:
                        if not state.environment_pending or not state.active:
                            return
                        state.environment_pending = False
                        if state.voice_pending:
                            state.environment_pending = True
                            retry = True
                        else:
                            retry = False
                            revision = state.revision
                            state.trim_history()
                            candidate = deepcopy(state.backend)
                            candidate.update_environment(state.context_with_conversation())
                    if not retry:
                        decision = candidate.evaluate_environment()
                        with state.lock:
                            if revision == state.revision and state.active and not state.voice_pending:
                                if decision['action'] == 'speak':
                                    state.outbox = {**decision, 'revision': revision,
                                                    't0': state.context['t0'], 'user': user}
                                print(f"ENVIRONMENT {user} {candidate.t0}: {decision['action']}", flush=True)
                            elif state.active:
                                state.environment_pending = True
                if retry:
                    time.sleep(0.05)
        except Exception:
            logging.exception('Evaluación del entorno fallida para %s', user)
        finally:
            with state.lock:
                state.worker_running = False
                # Una actualización puede llegar justo antes de salir del bucle.
                if state.environment_pending and state.active:
                    state.worker_running = True
                    Thread(target=self._evaluate, args=(user, state), daemon=True).start()

    def events(self, payload):
        _, state = self._session(payload)
        with state.lock:
            return deepcopy(state.outbox) if state.active and state.outbox else {'action': 'silent', 'text': ''}

    def spoken(self, payload):
        _, state = self._session(payload)
        with state.lock:
            event = state.outbox
            if state.voice_pending or not state.active or not event or event['revision'] != payload.get('revision'):
                return {'accepted': False}
            state.backend.messages.append({'role': 'assistant', 'content': event['text']})
            state.timeline.append({'timestamp': event['t0'], 'source': 'assistant', 'text': event['text']})
            state.outbox = None
            state.revision += 1
            return {'accepted': True}


class LiveSession:
    def __init__(self, backend):
        self.backend = backend
        self.lock, self.generation_lock = RLock(), RLock()
        self.context = build_context(backend.t0, backend.data_dir, history_hours=36)
        self.turns = deque()
        self.timeline = deque(maxlen=48)
        self.revision = self.voice_pending = 0
        self.active = self.environment_pending = self.worker_running = False
        self.outbox = None

    def context_with_conversation(self):
        now = time.monotonic()
        while self.turns and self.turns[0] <= now - 60:
            self.turns.popleft()
        return {**deepcopy(self.context), 'user_turns_last_minute': len(self.turns),
                'conversation_active': self.active,
                'recent_conversation_timeline': list(self.timeline)}

    def trim_history(self):
        # Recortar en límites de turno para no dejar resultados de tools huérfanos.
        starts = [i for i, message in enumerate(self.backend.messages) if message['role'] == 'user']
        if len(starts) > 24:
            self.backend.messages[1:starts[-24]] = []


class RemoteConversation:
    def __init__(self, endpoint, t0, user, model, session_id='reachy'):
        self.endpoint, self.t0, self.user = endpoint, t0, user_id(user)
        self.model, self.session_id, self.last_gesture = model, session_id, None

    def _post(self, path, extra=None):
        from .environment_service import endpoint_url, post
        payload = dict(t0=self.t0, user=self.user, model=self.model, session_id=self.session_id)
        payload.update(extra or {})
        return post(endpoint_url(self.endpoint, path), payload)

    def activate(self):
        result = self._post('/activate')
        self.t0 = result['t0']

    def deactivate(self):
        self._post('/deactivate')

    def poll(self):
        return self._post('/events')

    def accept(self, event):
        return self._post('/spoken', {'revision': event['revision']})['accepted']

    def respond(self, text, trace=None):
        import json
        from urllib.request import Request, urlopen
        payload = dict(text=text, t0=self.t0, user=self.user, model=self.model,
                       session_id=self.session_id)
        with urlopen(Request(self.endpoint, data=json.dumps(payload).encode(),
                            headers={'Content-Type': 'application/json'}), timeout=180) as response:
            result = json.load(response)
        self.last_gesture = result.get('gesture')
        self.t0 = result.get('t0', self.t0)
        if trace:
            for item in result.get('trace', []):
                trace(item['event'], item['data'])
            trace('remote_response', result)
        return result['text']
