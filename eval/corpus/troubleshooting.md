# Troubleshooting

## Jobs stuck in the leased state

Workers send a heartbeat for every job they hold every 15 seconds. If the server hears
nothing for the lease timeout, 60 seconds by default, it assumes the worker died and puts
the job back in its queue. Jobs that stay `leased` far longer than that usually belong to
a worker that is alive but blocked; check its logs for a handler that never returns, and
set a per-job timeout.

## "database is locked" errors

SQLite allows only one writer at a time. Under heavy load, many workers reporting results
at once can produce `database is locked` errors. The fix is to move to the PostgreSQL
backend; raising timeouts only hides the problem.

## Worker memory keeps growing

Some handlers leak memory. A worker process restarts itself after it has run
`workers.max_jobs_per_process` jobs, 1000 by default, which caps the damage. Lower the
value if memory still grows too much.

## Clock skew warnings

The server compares its clock with each worker's on every heartbeat and logs a warning
when they differ by more than 2 seconds. Large skew makes schedules fire at the wrong
time; run NTP on every machine.
