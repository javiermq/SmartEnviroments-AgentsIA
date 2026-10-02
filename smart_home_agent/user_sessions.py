"""Historias aisladas por usuario y sesión, compartiendo el mismo Ollama."""
from pathlib import Path
from threading import RLock
from .context import DATA_DIR
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

    def respond(self, payload, trace=None):
        if not isinstance(payload, dict):
            raise ValueError('El cuerpo debe ser un objeto JSON')
        user = user_id(payload.get('user'))
        session = payload.get('session_id', 'reachy')
        if not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError('session_id inválido')
        text, t0 = payload.get('text'), payload.get('t0')
        if not isinstance(text, str) or not text.strip() or not isinstance(t0, str):
            raise ValueError('Se requieren text y t0')
        model = payload.get('model', 'qwen3:4b-instruct-2507-q4_K_M')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model inválido')
        with self.lock:
            key = (session, user, t0, model)
            if key not in self.sessions:
                self.sessions[key] = (ConversationBackend(t0, model, self.ollama_url,
                                                          self.data_root / user), RLock())
            backend, session_lock = self.sessions[key]
        with session_lock:
            events = []
            def record(event, data):
                events.append({'event': event, 'data': data})
                if trace:
                    trace(event, data)
            answer = backend.respond(text, record)
            return {'text': answer, 'gesture': backend.last_gesture, 'user': user,
                    'trace': events}


class RemoteConversation:
    def __init__(self, endpoint, t0, user, model, session_id='reachy'):
        self.endpoint, self.t0, self.user = endpoint, t0, user_id(user)
        self.model, self.session_id, self.last_gesture = model, session_id, None

    def respond(self, text, trace=None):
        import json
        from urllib.request import Request, urlopen
        payload = dict(text=text, t0=self.t0, user=self.user, model=self.model,
                       session_id=self.session_id)
        with urlopen(Request(self.endpoint, data=json.dumps(payload).encode(),
                            headers={'Content-Type': 'application/json'}), timeout=180) as response:
            result = json.load(response)
        self.last_gesture = result.get('gesture')
        if trace:
            for item in result.get('trace', []):
                trace(item['event'], item['data'])
            trace('remote_response', result)
        return result['text']
