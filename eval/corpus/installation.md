# Installation

## Requirements

Tidewater runs on 64-bit Linux and macOS. The server needs about 100 MB of memory plus
whatever the database uses. PostgreSQL 14 or newer is required when you use the
PostgreSQL backend. There are no other runtime dependencies.

## Linux

Download the release archive, unpack it and copy the binary onto your `PATH`:

```bash
tar -xzf tidewater-linux-amd64.tar.gz
sudo install -m 0755 tidewater /usr/local/bin/tidewater
```

To run the server as a service, install the bundled systemd unit with
`sudo tidewater service install`. The unit runs as the unprivileged `tidewater` user,
which the command creates if it does not exist.

## macOS

On macOS, install from the project's Homebrew tap with `brew install tidewater`. Homebrew
keeps the binary updated together with your other packages.

## Container image

A container image is published for every release. Mount a volume at `/var/lib/tidewater`
so that the SQLite database survives container restarts.

## Verifying the installation

Run `tidewater version` to print the version and build date, and `tidewater doctor` to
check that the configuration file parses and the database is reachable.

## Upgrading

Stop the server, replace the binary and start it again. Database migrations run
automatically on start-up. Migrations cannot be rolled back, so take a backup with
`tidewater backup create` before every upgrade. Workers may run one minor version behind
the server, which allows rolling upgrades of large worker fleets.
