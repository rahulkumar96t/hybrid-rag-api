# Tidewater overview

Tidewater is a fictional, self-hosted job queue used as the sample corpus for this
repository. It ships as a single static binary called `tidewater` that runs either as a
server or as a worker. Everything in these documents is invented for evaluation purposes.

## Core concepts

A **job** is a unit of work: a JSON payload plus the name of the handler that should
process it. Jobs are submitted to a **queue**. Each queue has a priority between 0 and
9; workers always lease jobs from higher-priority queues first.

A **worker** is a process that leases jobs, runs the handler and reports the outcome.
While a worker holds a job, the job is in the `leased` state. A **schedule** submits a
job automatically according to a cron expression.

## Architecture

The server stores all state in a single database. Two storage backends are supported:
SQLite for single-node installations and PostgreSQL for anything larger. Workers never
talk to the database directly; they use the server's HTTP API, so workers can run on
different machines from the server.

Job state transitions are `queued`, `leased`, then one of `succeeded`, `failed` or
`dead`. A job becomes `dead` when it has used all of its retry attempts and is moved to
the dead-letter queue.

## When not to use Tidewater

Tidewater is designed for background work measured in seconds to hours. It is not a
streaming platform: there is no ordering guarantee between jobs in the same queue, and
it is a poor fit for workloads that need more than a few thousand jobs per second. For
exactly-once processing, make handlers idempotent; Tidewater delivers jobs at least once.
