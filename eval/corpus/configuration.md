# Configuration

## Configuration sources

Settings are read from four places. Command-line flags win over environment variables,
environment variables win over the configuration file, and the configuration file wins
over built-in defaults. Environment variables use the `TIDEWATER_` prefix and upper-case
names, so `server.listen` becomes `TIDEWATER_SERVER_LISTEN`.

The configuration file is `/etc/tidewater/tidewater.toml` by default. Point to another
file with the `--config` flag.

## Server

`server.listen` sets the address the HTTP API binds to. The default is `127.0.0.1:7420`,
which only accepts local connections; set it to `0.0.0.0:7420` to accept remote workers.
`server.request_timeout` limits how long a single API request may take (default 30 s).

## Storage

`storage.backend` is either `sqlite` (the default) or `postgres`. For SQLite,
`storage.path` sets the database file, by default `/var/lib/tidewater/tidewater.db`.
For PostgreSQL, set `storage.dsn` to a connection string; the server keeps a pool of at
most `storage.max_connections` connections (default 20).

## Workers

`workers.concurrency` is the number of jobs a single worker process runs at the same
time. It defaults to the number of CPU cores. `workers.queues` lists the queues a worker
leases from; an empty list means all queues.

## Retention

Finished jobs are purged by a background task that runs every hour. Jobs that succeeded
are kept for 7 days and failed or dead jobs for 30 days. Change this with
`retention.succeeded` and `retention.failed`, which accept durations such as `72h`.
Setting a value to `0` keeps jobs forever, which makes the database grow without bound.
