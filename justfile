build:
    cargo build

check:
    cargo fmt --check
    cargo clippy --all --benches --tests --examples --all-features
    cargo test

discover:
    cargo run -- discover --config wiimi-site.toml

site:
    cargo run -- site --config wiimi-site.toml

# Mirror the CI workflow locally with a small 3-image config
site-local: build
    cargo run -- discover --config wiimi-site-local.toml --db local-scans.db
    cargo run -- site --config wiimi-site-local.toml --db local-scans.db
    open _site/index.html

# Deploy _site/ to Cloudflare Pages (requires `wrangler` and CF auth)
deploy:
    wrangler pages deploy _site --project-name=wiimi --branch=main

# Build the full site and deploy to Cloudflare Pages
deploy-full: build discover site deploy

# Build the local preview site and deploy to Cloudflare Pages
deploy-local: site-local deploy
