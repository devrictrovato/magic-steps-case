"""
root_route.py — Rota "/" da Magic Steps API

Adicionar ao routes.py:

    from .root_route import root_page
    router.add_api_route("/", root_page, methods=["GET"], include_in_schema=False)

Ou simplesmente copiar o endpoint `root_page` para dentro de routes.py.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .context import get_model_context

router = APIRouter()

_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Magic Steps API</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet" />

  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:        #0a0c10;
      --surface:   #0f1318;
      --border:    #1e2530;
      --accent:    #3be8a0;
      --accent2:   #1a7fff;
      --warn:      #f5a623;
      --danger:    #ff4d6d;
      --text:      #e2e8f0;
      --muted:     #64748b;
      --mono:      'DM Mono', monospace;
      --sans:      'Syne', sans-serif;
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ── grid de fundo ── */
    body::before {{
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(59,232,160,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,232,160,.035) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }}

    /* ── glow de fundo ── */
    .bg-glow {{
      position: fixed;
      top: -200px;
      left: 50%;
      transform: translateX(-50%);
      width: 900px;
      height: 500px;
      background: radial-gradient(ellipse at center,
        rgba(59,232,160,.08) 0%,
        rgba(26,127,255,.05) 40%,
        transparent 70%);
      pointer-events: none;
      z-index: 0;
    }}

    /* ── layout ── */
    .wrap {{
      position: relative;
      z-index: 1;
      max-width: 960px;
      margin: 0 auto;
      padding: 0 24px 80px;
    }}

    /* ── header ── */
    header {{
      padding: 64px 0 48px;
      animation: fadeDown .6s ease both;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 500;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--accent);
      background: rgba(59,232,160,.08);
      border: 1px solid rgba(59,232,160,.2);
      padding: 4px 12px;
      border-radius: 100px;
      margin-bottom: 24px;
    }}

    .badge::before {{
      content: '';
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 2s ease infinite;
    }}

    h1 {{
      font-size: clamp(2.4rem, 5vw, 3.8rem);
      font-weight: 800;
      letter-spacing: -.03em;
      line-height: 1.05;
      margin-bottom: 16px;
    }}

    h1 span {{ color: var(--accent); }}

    .subtitle {{
      font-family: var(--mono);
      font-size: .9rem;
      font-weight: 300;
      color: var(--muted);
      max-width: 560px;
      line-height: 1.7;
    }}

    /* ── status bar ── */
    .status-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 40px 0;
      animation: fadeUp .5s .15s ease both;
    }}

    .status-pill {{
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 16px;
      font-family: var(--mono);
      font-size: .8rem;
    }}

    .dot {{
      width: 7px; height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    .dot.green  {{ background: var(--accent); box-shadow: 0 0 8px var(--accent); }}
    .dot.red    {{ background: var(--danger); box-shadow: 0 0 8px var(--danger); }}
    .dot.yellow {{ background: var(--warn);   box-shadow: 0 0 8px var(--warn); }}

    .pill-label {{ color: var(--muted); margin-right: 2px; }}
    .pill-value {{ color: var(--text); font-weight: 500; }}

    /* ── seções ── */
    section {{
      margin-top: 56px;
      animation: fadeUp .5s .3s ease both;
    }}

    .section-title {{
      font-family: var(--mono);
      font-size: .7rem;
      font-weight: 500;
      letter-spacing: .15em;
      text-transform: uppercase;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
      margin-bottom: 24px;
    }}

    /* ── grid de rotas ── */
    .routes-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }}

    .route-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      transition: border-color .2s, transform .2s;
      cursor: default;
    }}

    .route-card:hover {{
      border-color: rgba(59,232,160,.3);
      transform: translateY(-2px);
    }}

    .route-head {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }}

    .method {{
      font-family: var(--mono);
      font-size: .65rem;
      font-weight: 500;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}

    .method.GET    {{ background: rgba(59,232,160,.12); color: var(--accent); }}
    .method.POST   {{ background: rgba(26,127,255,.12); color: var(--accent2); }}
    .method.PUT    {{ background: rgba(245,166,35,.12);  color: var(--warn); }}

    .route-path {{
      font-family: var(--mono);
      font-size: .82rem;
      color: var(--text);
    }}

    .route-desc {{
      font-size: .8rem;
      color: var(--muted);
      line-height: 1.5;
    }}

    /* ── classes ── */
    .classes-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .class-card {{
      flex: 1;
      min-width: 160px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      text-align: center;
    }}

    .class-idx {{
      font-family: var(--mono);
      font-size: 2rem;
      font-weight: 300;
      color: var(--muted);
      display: block;
      margin-bottom: 6px;
    }}

    .class-label {{
      font-size: .95rem;
      font-weight: 700;
      letter-spacing: .04em;
      display: block;
      margin-bottom: 4px;
    }}

    .class-label.atraso  {{ color: var(--danger); }}
    .class-label.neutro  {{ color: var(--warn); }}
    .class-label.avanco  {{ color: var(--accent); }}

    .class-desc {{
      font-size: .75rem;
      color: var(--muted);
    }}

    /* ── pipeline ── */
    .pipeline {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0;
    }}

    .pipe-step {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 18px;
      font-family: var(--mono);
      font-size: .78rem;
    }}

    .pipe-step .step-num {{
      color: var(--accent);
      font-weight: 500;
      display: block;
      margin-bottom: 2px;
      font-size: .65rem;
      letter-spacing: .1em;
      text-transform: uppercase;
    }}

    .pipe-step .step-name {{ color: var(--text); }}

    .pipe-arrow {{
      color: var(--muted);
      font-size: 1.1rem;
      padding: 0 8px;
    }}

    /* ── links ── */
    .links-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .link-btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 12px 22px;
      border-radius: 8px;
      font-family: var(--mono);
      font-size: .82rem;
      font-weight: 500;
      text-decoration: none;
      transition: opacity .2s, transform .2s;
    }}

    .link-btn:hover {{ opacity: .85; transform: translateY(-1px); }}

    .link-btn.primary {{
      background: var(--accent);
      color: #0a0c10;
    }}

    .link-btn.secondary {{
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
    }}

    /* ── footer ── */
    footer {{
      margin-top: 72px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      font-family: var(--mono);
      font-size: .75rem;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
      animation: fadeUp .5s .4s ease both;
    }}

    /* ── animações ── */
    @keyframes fadeDown {{
      from {{ opacity:0; transform: translateY(-16px); }}
      to   {{ opacity:1; transform: translateY(0); }}
    }}
    @keyframes fadeUp {{
      from {{ opacity:0; transform: translateY(16px); }}
      to   {{ opacity:1; transform: translateY(0); }}
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity:1; transform: scale(1); }}
      50%       {{ opacity:.5; transform: scale(.8); }}
    }}
  </style>
</head>
<body>
  <div class="bg-glow"></div>

  <div class="wrap">

    <header>
      <div class="badge">PEDE 2024 · v{version}</div>
      <h1>Magic <span>Steps</span><br/>Prediction API</h1>
      <p class="subtitle">
        API de inferência para predição de defasagem escolar.<br/>
        Classificação multiclasse com Deep Learning + pipeline automático de pré-processamento.
      </p>
    </header>

    <!-- STATUS -->
    <div class="status-bar">
      <div class="status-pill">
        <span class="dot {model_dot}"></span>
        <span class="pill-label">modelo</span>
        <span class="pill-value">{model_status}</span>
      </div>
      <div class="status-pill">
        <span class="dot {prep_dot}"></span>
        <span class="pill-label">preprocessador</span>
        <span class="pill-value">{prep_status}</span>
      </div>
      <div class="status-pill">
        <span class="dot {fe_dot}"></span>
        <span class="pill-label">feature engineering</span>
        <span class="pill-value">{fe_status}</span>
      </div>
      <div class="status-pill">
        <span class="dot green"></span>
        <span class="pill-label">device</span>
        <span class="pill-value">{device}</span>
      </div>
      <div class="status-pill">
        <span class="dot green"></span>
        <span class="pill-label">input dim</span>
        <span class="pill-value">{input_dim}</span>
      </div>
    </div>

    <!-- DOCUMENTAÇÃO -->
    <section>
      <div class="section-title">Documentação</div>
      <div class="links-row">
        <a class="link-btn primary" href="/docs">⚡ Swagger UI</a>
        <a class="link-btn secondary" href="/redoc">📄 ReDoc</a>
        <a class="link-btn secondary" href="/openapi.json">&#123;&#125; OpenAPI JSON</a>
        <a class="link-btn secondary" href="/health">💚 Health Check</a>
      </div>
    </section>

    <!-- ROTAS -->
    <section>
      <div class="section-title">Rotas disponíveis</div>
      <div class="routes-grid">

        <div class="route-card">
          <div class="route-head">
            <span class="method GET">GET</span>
            <span class="route-path">/features</span>
          </div>
          <p class="route-desc">Lista as 16 features de entrada com tipo, intervalo e categorias válidas.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method POST">POST</span>
            <span class="route-path">/predict</span>
          </div>
          <p class="route-desc">Prediz a defasagem de um único aluno. Retorna classe, label e probabilidades.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method POST">POST</span>
            <span class="route-path">/predict/batch</span>
          </div>
          <p class="route-desc">Predição em lote — até 500 alunos por chamada.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method POST">POST</span>
            <span class="route-path">/features/process</span>
          </div>
          <p class="route-desc">Executa o pipeline e retorna cada etapa: raw → normalized → engineered.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method GET">GET</span>
            <span class="route-path">/thresholds</span>
          </div>
          <p class="route-desc">Consulta o threshold de confiança atual.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method PUT">PUT</span>
            <span class="route-path">/thresholds</span>
          </div>
          <p class="route-desc">Atualiza o threshold de confiança (0.0 – 1.0).</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method GET">GET</span>
            <span class="route-path">/monitor/logs</span>
          </div>
          <p class="route-desc">Histórico de predições com filtros por usuário e student_id.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method GET">GET</span>
            <span class="route-path">/monitor/drift</span>
          </div>
          <p class="route-desc">Resumo de distribuição de classes — detecção de drift.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method GET">GET</span>
            <span class="route-path">/monitor/runs</span>
          </div>
          <p class="route-desc">Lista todos os runs de treinamento registrados no PostgreSQL.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method GET">GET</span>
            <span class="route-path">/monitor/events</span>
          </div>
          <p class="route-desc">Eventos de monitoramento com filtros por tipo e severidade.</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method POST">POST</span>
            <span class="route-path">/token</span>
          </div>
          <p class="route-desc">Gera um JWT Bearer token (OAuth2 Password Flow).</p>
        </div>

        <div class="route-card">
          <div class="route-head">
            <span class="method POST">POST</span>
            <span class="route-path">/register</span>
          </div>
          <p class="route-desc">Registra um novo usuário na API.</p>
        </div>

      </div>
    </section>

    <!-- PIPELINE -->
    <section>
      <div class="section-title">Pipeline de inferência</div>
      <div class="pipeline">
        <div class="pipe-step">
          <span class="step-num">01</span>
          <span class="step-name">16 features brutas</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step">
          <span class="step-num">02</span>
          <span class="step-name">ColumnTransformer</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step">
          <span class="step-num">03</span>
          <span class="step-name">Feature Engineering (+7)</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step">
          <span class="step-num">04</span>
          <span class="step-name">MagicStepsNet ({input_dim}d)</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step">
          <span class="step-num">05</span>
          <span class="step-name">Softmax → classe</span>
        </div>
      </div>
    </section>

    <!-- CLASSES -->
    <section>
      <div class="section-title">Classes de saída</div>
      <div class="classes-row">
        <div class="class-card">
          <span class="class-idx">0</span>
          <span class="class-label atraso">atraso</span>
          <span class="class-desc">Aluno com defasagem — abaixo do esperado para a fase</span>
        </div>
        <div class="class-card">
          <span class="class-idx">1</span>
          <span class="class-label neutro">neutro</span>
          <span class="class-desc">Aluno dentro do esperado para a fase</span>
        </div>
        <div class="class-card">
          <span class="class-idx">2</span>
          <span class="class-label avanco">avanço</span>
          <span class="class-desc">Aluno acima do esperado para a fase</span>
        </div>
      </div>
    </section>

    <footer>
      <span>Magic Steps API · Projeto PEDE 2024</span>
      <span>versão {version} · {device}</span>
    </footer>

  </div>
</body>
</html>"""


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Página inicial informativa da API."""
    ctx = get_model_context()

    model_loaded = ctx["model"] is not None
    prep_loaded  = ctx["preprocessor"] is not None
    uses_fe      = ctx.get("uses_feature_engineering", False)
    input_dim    = ctx.get("input_dim")
    version      = "2.1.0"

    html = _HTML.format(
        version      = version,
        device       = str(ctx["device"]),
        input_dim    = str(input_dim) if input_dim else "—",
        model_status = "carregado" if model_loaded else "não carregado",
        model_dot    = "green" if model_loaded else "red",
        prep_status  = "carregado" if prep_loaded else "não carregado",
        prep_dot     = "green" if prep_loaded else "red",
        fe_status    = "ativo" if uses_fe else "inativo",
        fe_dot       = "green" if uses_fe else "yellow",
    )

    return HTMLResponse(content=html, status_code=200)