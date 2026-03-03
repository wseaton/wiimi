use std::sync::LazyLock;

use minijinja::Environment;

// HTML templates
const HTML_TEMPLATE: &str = include_str!("templates/html_template.html");
const DIFF_TEMPLATE: &str = include_str!("templates/diff_template.html");
const INDEX_TEMPLATE: &str = include_str!("templates/site_index_template.html");

// SVG OG card templates
const OG_SCAN: &str = include_str!("templates/og_card_scan.svg");
const OG_DIFF: &str = include_str!("templates/og_card_diff.svg");
const OG_INDEX: &str = include_str!("templates/og_card_index.svg");

// Static assets (not registered as templates, just embedded)
pub const FAVICON_SVG: &str = include_str!("templates/favicon.svg");
pub const BASE_CSS: &str = include_str!("templates/base_style.css");

/// HTML template environment (auto-escapes HTML).
pub static HTML_ENV: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::Html);
    env.add_template("html", HTML_TEMPLATE)
        .expect("html template");
    env.add_template("diff", DIFF_TEMPLATE)
        .expect("diff template");
    env.add_template("index", INDEX_TEMPLATE)
        .expect("index template");
    env
});

/// SVG template environment (no auto-escape, uses `svg_escape` filter).
pub static SVG_ENV: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    env.add_filter("svg_escape", |v: String| -> String {
        v.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    });
    env.add_template("scan", OG_SCAN).expect("og scan template");
    env.add_template("diff", OG_DIFF).expect("og diff template");
    env.add_template("index", OG_INDEX)
        .expect("og index template");
    env
});
