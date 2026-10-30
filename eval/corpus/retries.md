# Retries and failure handling

## Retry policy

When a handler fails, Tidewater retries the job. Each job may be attempted at most
`max_attempts` times; the default is 5 attempts including the first one. The policy can
be set per queue in the configuration file or per job when it is submitted.

## Backoff

The delay before the next attempt grows exponentially: it starts at 2 seconds and doubles
after every failed attempt, up to a cap of 10 minutes. Tidewater applies full jitter, so
the actual delay is a random value between zero and the computed delay. Jitter prevents
many failed jobs from being retried at exactly the same moment after an outage.

## Retryable and permanent errors

Not every failure deserves a retry. A handler signals a temporary failure by exiting with
code 75, which is always retried. Any other non-zero exit code is also retried unless the
handler writes `permanent: true` in its result, in which case the job fails immediately
without using its remaining attempts. Timeouts count as temporary failures.

## Dead-letter queue

A job that has used all of its attempts moves to the dead-letter queue and its state
becomes `dead`. Dead jobs are never retried automatically. Inspect them with
`tidewater dlq list` and send one back to its original queue with
`tidewater dlq requeue <job-id>`, which resets its attempt counter.
