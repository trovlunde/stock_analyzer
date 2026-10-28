# ADR: Flask vs FastAPI for the web API

- Status: Accepted
- Related: [#13](https://github.com/trovlunde/stock_analyzer/issues/13), [stack-improvement-plan.md](./stack-improvement-plan.md)

## Context

`stock_analysis/web/app.py` exposes 10 routes (index, health, status, model
listing, predict, cache clear, train, movement predict, movement model list,
movement-analysis page) on Flask + `flask-cors`, with two Jinja2 templates
rendered via `render_template`. Every route does synchronous, CPU-bound work
(sklearn training/inference, pandas transforms, in-process `ModelManager`
cache) — none awaits I/O concurrently or needs websockets/streaming.

The question: migrate to FastAPI for async support, Pydantic validation, and
auto-generated OpenAPI docs, or stay on Flask?

## Decision

**Stay on Flask.** No migration.

## Rationale

- **No concurrency need.** FastAPI's main advantage — async I/O — buys
  nothing here. Training/inference routes hold the GIL on CPU-bound sklearn
  calls regardless of framework; async wouldn't let other requests interleave
  productively on a single worker, and this service isn't run with multiple
  async workers today.
- **Small surface.** 10 routes is well within what either framework handles
  equally well. A rewrite is pure churn for no capability gain.
- **Template rendering stays simple.** Two Jinja2 templates render natively
  via Flask's `render_template`. FastAPI has no first-class server-rendered
  HTML story — it would need `Jinja2Templates` wired by hand, trading a
  built-in for an equivalent amount of code.
- **Request validation is already adequate.** Routes do manual
  `request.get_json()` / `request.args.get()` checks with explicit 400s.
  Pydantic schemas would be nicer, but 10 routes don't justify introducing a
  new dependency and request/response model layer.
- **No OpenAPI consumer today.** Nothing currently reads or needs an
  auto-generated OpenAPI spec; it's a nice-to-have, not a requirement.

## Revisit when

- The route count grows materially beyond today's 10, or
- A concrete need for async I/O (e.g. concurrent calls to external data
  providers inside a single request) or WebSocket/streaming responses shows
  up, or
- Automated API clients need a generated OpenAPI/Swagger contract.

None of these are true today, matching the existing recommendation in
[stack-improvement-plan.md](./stack-improvement-plan.md#33-api-modernisation--13).
