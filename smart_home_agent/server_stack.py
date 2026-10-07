"""Arranca voz, conversor y entorno como procesos separados en una consola."""
import argparse
import os
import subprocess
import sys
import time
from urllib.request import urlopen


def wait_ready(process, url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'El servicio terminó con código {process.returncode}: {url}')
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f'El servicio no respondió: {url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--voice-port', type=int, default=11435)
    parser.add_argument('--conversation-port', type=int, default=11437)
    parser.add_argument('--t0', required=True)
    parser.add_argument('--session-id', default='reachy')
    parser.add_argument('--model', default='qwen3:4b-instruct-2507-q4_K_M')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434/api/chat')
    parser.add_argument('--tts-model', required=True)
    parser.add_argument('--stt-model', default='turbo')
    parser.add_argument('--stt-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--period', type=float, default=60)
    args = parser.parse_args()
    processes = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

    def launch(module, options):
        process = subprocess.Popen([sys.executable, '-u', '-m', module, *options],
                                   creationflags=flags)
        processes.append(process)
        return process

    connect_host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    conversation_url = f'http://{connect_host}:{args.conversation_port}'
    try:
        conversation = launch('smart_home_agent.conversation_service', [
            '--host', args.host, '--port', str(args.conversation_port), '--ollama-url', args.ollama_url])
        wait_ready(conversation, conversation_url + '/health')
        voice = launch('smart_home_agent.human_console_bridge', [
            '--service', 'voice', '--host', args.host, '--port', str(args.voice_port),
            '--tts-model', args.tts_model, '--stt-model', args.stt_model, '--stt-device', args.stt_device])
        wait_ready(voice, f'http://{connect_host}:{args.voice_port}/health')
        environment = launch('smart_home_agent.environment_service', [
            '--conversation-url', conversation_url + '/conversation', '--t0', args.t0,
            '--session-id', args.session_id, '--model', args.model, '--period', str(args.period)])
        print('Voz, conversor y entorno activos. Ctrl+C los detiene.', flush=True)
        while True:
            if conversation.poll() is not None or voice.poll() is not None:
                raise RuntimeError('Un servicio terminó inesperadamente')
            if environment.poll() is not None:
                if environment.returncode:
                    raise RuntimeError('El servicio de entorno terminó con error')
                # Al agotarse los datos, voz y sesiones siguen disponibles.
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
