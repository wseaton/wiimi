/// Shared gruvbox CSS: variables, reset, body font, and masthead tabs.
/// Both the scan and diff HTML templates inject this at `/*BASE_STYLES*/`.
pub const BASE_CSS: &str = include_str!("base_style.css");

/// Controls whether JS dependencies are inlined or loaded from a CDN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleMode {
    /// Inline all JS into the HTML (default, fully offline-capable).
    SelfContained,
    /// Use `<script src="...">` tags pointing at unpkg.com.
    Cdn,
}

const CYTOSCAPE_JS: &str = include_str!("vendor/cytoscape.min.js");
const DAGRE_JS: &str = include_str!("vendor/dagre.min.js");
const CYTOSCAPE_DAGRE_JS: &str = include_str!("vendor/cytoscape-dagre.js");
const CYTOSCAPE_NODE_HTML_LABEL_JS: &str = include_str!("vendor/cytoscape-node-html-label.min.js");

/// Returns the `<script>` block to splice into the HTML template at `<!--SCRIPTS-->`.
pub fn script_block(mode: BundleMode) -> String {
    match mode {
        BundleMode::SelfContained => {
            format!(
                "<script>{CYTOSCAPE_JS}</script>\n\
                 <script>{DAGRE_JS}</script>\n\
                 <script>{CYTOSCAPE_DAGRE_JS}</script>\n\
                 <script>{CYTOSCAPE_NODE_HTML_LABEL_JS}</script>"
            )
        }
        BundleMode::Cdn => {
            r#"<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<script src="https://unpkg.com/cytoscape-node-html-label@1.2.2/dist/cytoscape-node-html-label.min.js"></script>"#
                .to_string()
        }
    }
}

/// Base64-encode bytes (thin wrapper around the `base64` crate).
pub fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use crate::style::{script_block, BundleMode};

    #[test]
    fn self_contained_inlines_js() {
        let block = script_block(BundleMode::SelfContained);
        assert!(
            block.contains("cytoscape"),
            "self-contained block should contain inline cytoscape JS"
        );
        assert!(
            !block.contains("unpkg.com"),
            "self-contained block should not reference CDN"
        );
    }

    #[test]
    fn cdn_uses_script_src() {
        let block = script_block(BundleMode::Cdn);
        assert!(
            block.contains(r#"<script src="https://unpkg.com/cytoscape@3.30.4"#),
            "CDN block should contain script src tags"
        );
        assert!(
            block.contains("unpkg.com"),
            "CDN block should reference unpkg.com"
        );
    }
}
