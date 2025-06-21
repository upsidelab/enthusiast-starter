<div align="center">
  <a href="https://upsidelab.io/tools/enthusiast" />
    <img src="https://github.com/user-attachments/assets/966204c3-ff69-47b2-a247-9f9cfa4e5b7d" height="150px" alt="Enthusiast">
  </a>
</div>

<h1 align="center">enthusiast.</h1>

<p align="center">Your open-souce AI agent for e-commerce.</p>
<div align="center">
  <strong>
    <a href="https://upsidelab.io/tools/enthusiast/docs/getting-started/installation">Get Started</a> |
    <a href="https://upsidelab.io/tools/enthusiast/docs">View Docs</a> |
    <a href="https://upsidelab.io/tools/enthusiast">Website</a>
  </strong>
</div>

## Introduction
Enthusiast is your open-source AI agent for e-commerce. Connect it to your product catalog, add content that describes your products and brand, and scale your team’s capabilities.

## Starter Pack

This repository provides everything you need to build custom agents and integrations, and deploy them to production with ease.

### Building a custom agent

Use the `src/` directory for your code—create agents or plugins using the interfaces defined in `enthusiast-common`, and enable them via `settings_override.py`.

The included Dockerfile installs Enthusiast, builds your custom package, and configures the system to use it.

To bootstrap your environment variables for Docker, run:
```shell
cp config/env.sample config/env
``` 
to bootstrap environment variables for the docker setup.

Then start the application locally:
```shell
docker compose build && docker compose up
```
