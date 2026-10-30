# Security

## API tokens

Every API request must carry a bearer token. Tokens have one of three scopes: `read` can
list queues and jobs, `submit` can also enqueue jobs, and `admin` can additionally change
configuration, manage schedules and requeue dead jobs. Workers need the `submit` scope
because they report job results through the API.

Create a token with `tidewater token create --scope submit`. Tokens are shown once and
stored only as a salted hash.

## Rotating tokens

Tokens do not expire by default. To rotate a token, create the new token first, roll it
out to the clients, then revoke the old one with `tidewater token revoke <id>`. A revoked
token is rejected immediately.

## TLS

The server can terminate TLS itself: set `tls.cert_file` and `tls.key_file`. When both
are set, plain HTTP is refused. Certificates are reloaded when the files change, so a
renewal does not need a restart.

## Secrets in payloads

Job payloads are stored in the database in plain text and appear in the admin UI, so do
not put credentials in them. Use a secret reference of the form `secret://<name>` instead;
the worker resolves it from its local secret store just before the handler runs. Payload
fields whose names contain `password` or `token` are masked in logs.
