# Deploy

Host: **arya** (SSH `root` via Tailscale — use your SSH config, not a checked-in IP)  
App: systemd `queuescore` → `:8501`  
Edge: Cloudflare Tunnel → **queuescore.tech**  
CI deploy: self-hosted runner label `queuescore` on push to `main`

```
push main → runner → git pull /opt/queuescore → pip install → systemctl restart queuescore
```

| Path | What |
|------|------|
| `/opt/queuescore` | app + `.venv` |
| `/opt/actions-queuescore` | repo-scoped runner |
| `/etc/cloudflared/config.yml` | tunnel ingress |
| [deploy.yml](.github/workflows/deploy.yml) | auto-deploy |
| [queuescore.service](deploy/queuescore.service) | systemd unit |

### DNS (Cloudflare)

Point the zone’s nameservers at Cloudflare, then add **proxied** CNAMEs for `@` and `www` to your tunnel’s `*.cfargotunnel.com` target (from Zero Trust / `cloudflared tunnel info` — do not commit the UUID).

### Runner

```bash
REGISTRATION_TOKEN=… /opt/queuescore/scripts/setup_runner.sh
```

### Ops

```bash
ssh arya   # Host arya / User root in ~/.ssh/config
journalctl -u queuescore -f
/opt/queuescore/scripts/restart.sh
curl -I https://queuescore.tech
```
