# Observability

## Metrics

The server exposes Prometheus metrics at `/metrics`. The most useful series are
`tidewater_jobs_enqueued_total` (a counter per queue), `tidewater_job_duration_seconds`
(a histogram of handler run time) and `tidewater_queue_depth` (a gauge with the number of
queued jobs per queue). Alert on queue depth growing for longer than your normal batch
window rather than on a fixed number.

## Logs

Logs are written to standard output as JSON lines. Each line has a `level`, a timestamp
and, for job-related events, the job id and queue. The log level is `info` by default;
set `log.level` to `debug` while investigating a problem, because debug logging is
verbose.

## Tracing

Tidewater can export OpenTelemetry traces over OTLP. Set the
`OTEL_EXPORTER_OTLP_ENDPOINT` environment variable on the server and on workers; each job
then gets a span that covers its time in the queue and its run time.

## Health checks

`/healthz` returns 200 while the process is running. `/readyz` returns 200 only when the
database is reachable and migrations are complete, which makes it the right endpoint for
load-balancer and orchestrator readiness probes.
