# Scheduling recurring jobs

## Cron expressions

Schedules use standard five-field cron expressions: minute, hour, day of month, month and
day of week. An optional sixth field at the start adds seconds. The shortcuts `@hourly`,
`@daily` and `@weekly` are also accepted.

## Time zones

Schedules are evaluated in UTC unless the schedule sets a `timezone` field with an IANA
name such as `Europe/Zurich`. During a daylight-saving change, a local time that does not
exist is skipped and a local time that occurs twice runs only once.

## Missed runs

If the server was down when a schedule should have fired, the `catch_up` setting decides
what happens on restart. `none` skips the missed runs, `one` submits a single job for all
of them together, and `all` submits one job per missed run. The default is `one`.

## Overlapping runs

A schedule can fire while the previous run is still going. The `overlap` setting chooses
between `skip` (do not submit a new job), `queue` (submit it anyway; it waits for a free
worker) and `allow`. The default is `skip`.

## Pausing a schedule

`tidewater schedule pause <name>` stops a schedule from submitting jobs without deleting
it; `tidewater schedule resume <name>` turns it back on. Runs missed while paused are
never caught up.
