import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = """<!doctype html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta name=\"theme-color\" content=\"#edf4f0\">
  <title>RailroadDES | Visao geral</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18342f;
      --muted: #526963;
      --line: #cbd9d2;
      --paper: #f6f8f5;
      --green: #176b53;
      --orange: #d66b45;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background-color: var(--paper);
      background-image: linear-gradient(#dce8e1 1px, transparent 1px), linear-gradient(90deg, #dce8e1 1px, transparent 1px);
      background-size: 36px 36px;
      font: 16px/1.6 system-ui, sans-serif;
    }
    main {
      width: min(100% - 40px, 960px);
      margin: 56px auto;
      background: var(--paper);
      border: 1px solid var(--line);
    }
    header { padding: 32px clamp(24px, 6vw, 64px) 38px; border-bottom: 1px solid var(--line); }
    .eyebrow { color: var(--green); font: 700 12px/1.4 monospace; text-transform: uppercase; }
    h1, h2 { font-family: Georgia, serif; font-weight: 500; line-height: 1.15; }
    h1 { max-width: 650px; margin: 18px 0 12px; font-size: clamp(36px, 7vw, 62px); }
    header p { max-width: 680px; margin: 0; color: var(--muted); font-size: 18px; }
    .status { display: inline-flex; align-items: center; gap: 9px; margin-top: 26px; color: var(--green); font: 600 13px monospace; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--orange); }
    .dot.ready { background: var(--green); }
    section { padding: 28px clamp(24px, 6vw, 64px); border-bottom: 1px solid var(--line); }
    h2 { margin: 0 0 18px; font-size: 27px; }
    .columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
    article { border-top: 3px solid var(--green); padding-top: 12px; }
    article:nth-child(2) { border-color: var(--orange); }
    article h3 { margin: 0 0 6px; font-size: 15px; }
    article p, section > p { margin: 0; color: var(--muted); }
    footer { padding: 18px clamp(24px, 6vw, 64px); color: var(--muted); font: 12px monospace; }
    @media (max-width: 640px) {
      main { width: min(100% - 24px, 960px); margin: 12px auto; }
      .columns { grid-template-columns: 1fr; gap: 20px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class=\"eyebrow\">RailroadDES / simulacao ferroviaria</div>
      <h1>Uma ferrovia, modelada evento a evento.</h1>
      <p>Simulador para estudar o transporte de cargas, as operacoes dos trens e as decisoes de roteamento em uma rede ferroviaria.</p>
      <div class=\"status\"><span class=\"dot\" id=\"status-dot\"></span><span id=\"status-text\">Verificando servidor...</span></div>
    </header>
    <section>
      <h2>O que o modelo acompanha</h2>
      <div class=\"columns\">
        <article><h3>Movimento dos trens</h3><p>Viagens entre nos, trechos, filas e horarios de chegada e partida.</p></article>
        <article><h3>Operacoes e recursos</h3><p>Carregamento, descarregamento, capacidade dos pontos e disponibilidade de estoque.</p></article>
        <article><h3>Roteamento</h3><p>Escolha de fluxos e comparacao de estrategias de controle e otimizacao.</p></article>
      </div>
    </section>
    <section>
      <h2>Analise</h2>
      <p>As execucoes podem ser avaliadas por volume operado, cronogramas dos trens e niveis de estoque. Esta pagina e uma porta de entrada leve; ela nao inicia uma simulacao.</p>
    </section>
    <footer>RailroadDES · servidor HTTP local · <a href=\"/health\">status do servico</a></footer>
  </main>
  <script>
    fetch('/health').then(response => {
      if (!response.ok) throw new Error('health check failed');
      return response.json();
    }).then(() => {
      document.getElementById('status-dot').classList.add('ready');
      document.getElementById('status-text').textContent = 'Servidor online';
    }).catch(() => {
      document.getElementById('status-text').textContent = 'Status indisponivel';
    });
  </script>
</body>
</html>""".encode("utf-8")


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/health":
            response = json.dumps({"status": "ok", "service": "RailroadDES"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        else:
            self.send_error(404, "Pagina nao encontrada")


def main():
    parser = argparse.ArgumentParser(description="Inicia a pagina web local do RailroadDES.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface de rede (padrao: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Porta HTTP (padrao: 8000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"RailroadDES disponivel em http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()