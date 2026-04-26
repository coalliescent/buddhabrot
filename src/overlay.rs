//! Text overlays drawn via `fontdue` into the RGBA framebuffer.

use std::collections::HashMap;

use fontdue::{Font, FontSettings};

const FONT_BYTES: &[u8] = include_bytes!("../assets/font.ttf");

pub struct TextRenderer {
    font: Font,
    cache: HashMap<(char, u32), CachedGlyph>,
}

struct CachedGlyph {
    bitmap: Vec<u8>,   // coverage 0..255
    w: u32,
    h: u32,
    xmin: i32,
    ymin: i32,         // fontdue y-min (baseline-relative)
    advance: f32,
}

impl TextRenderer {
    pub fn new() -> Self {
        let font = Font::from_bytes(FONT_BYTES, FontSettings::default())
            .expect("bundled font failed to load");
        Self { font, cache: HashMap::new() }
    }

    fn glyph(&mut self, ch: char, px: u32) -> &CachedGlyph {
        self.cache.entry((ch, px)).or_insert_with(|| {
            let (metrics, bitmap) = self.font.rasterize(ch, px as f32);
            CachedGlyph {
                bitmap,
                w: metrics.width as u32,
                h: metrics.height as u32,
                xmin: metrics.xmin,
                ymin: metrics.ymin,
                advance: metrics.advance_width,
            }
        })
    }

    pub fn line_height(&self, px: u32) -> u32 {
        let m = self.font.horizontal_line_metrics(px as f32).unwrap();
        (m.new_line_size.ceil()) as u32
    }

    pub fn draw_sized(
        &mut self,
        frame: &mut [u8],
        fw: u32,
        fh: u32,
        text: &str,
        x: i32,
        y_baseline: i32,
        px: u32,
        color: [u8; 3],
    ) {
        let mut pen_x = x as f32;
        for ch in text.chars() {
            let meta = self.glyph(ch, px).clone_meta();
            let gb = self.glyph(ch, px);
            blit_coverage(
                frame, fw, fh,
                &gb.bitmap, gb.w, gb.h,
                (pen_x + gb.xmin as f32) as i32,
                y_baseline - gb.ymin - gb.h as i32,
                color,
            );
            pen_x += meta.advance;
        }
    }

    /// Width of a string in pixels at the given size.
    pub fn measure(&mut self, text: &str, px: u32) -> u32 {
        let mut w = 0.0f32;
        for ch in text.chars() {
            let g = self.glyph(ch, px);
            w += g.advance;
        }
        w.ceil() as u32
    }
}

impl CachedGlyph {
    fn clone_meta(&self) -> GlyphMeta {
        GlyphMeta {
            advance: self.advance,
            xmin: self.xmin,
            ymin: self.ymin,
            w: self.w,
            h: self.h,
        }
    }
}

struct GlyphMeta {
    advance: f32,
    #[allow(dead_code)] xmin: i32,
    #[allow(dead_code)] ymin: i32,
    #[allow(dead_code)] w: u32,
    #[allow(dead_code)] h: u32,
}

fn blit_coverage(
    frame: &mut [u8], fw: u32, fh: u32,
    bitmap: &[u8], gw: u32, gh: u32,
    x: i32, y: i32,
    color: [u8; 3],
) {
    if gw == 0 || gh == 0 { return; }
    for gy in 0..gh {
        let fy = y + gy as i32;
        if fy < 0 || fy as u32 >= fh { continue; }
        for gx in 0..gw {
            let fx = x + gx as i32;
            if fx < 0 || fx as u32 >= fw { continue; }
            let cov = bitmap[(gy * gw + gx) as usize] as u32;
            if cov == 0 { continue; }
            let idx = ((fy as u32 * fw + fx as u32) * 4) as usize;
            let a = cov as f32 / 255.0;
            for c in 0..3 {
                let dst = frame[idx + c] as f32;
                let src = color[c] as f32;
                frame[idx + c] = (dst + (src - dst) * a).clamp(0.0, 255.0) as u8;
            }
            frame[idx + 3] = 255;
        }
    }
}

/// Draw a translucent rectangle.
pub fn fill_rect(frame: &mut [u8], fw: u32, fh: u32, x: i32, y: i32, w: u32, h: u32, color: [u8; 4]) {
    for yy in 0..h as i32 {
        let fy = y + yy;
        if fy < 0 || fy as u32 >= fh { continue; }
        for xx in 0..w as i32 {
            let fx = x + xx;
            if fx < 0 || fx as u32 >= fw { continue; }
            let idx = ((fy as u32 * fw + fx as u32) * 4) as usize;
            let a = color[3] as f32 / 255.0;
            for c in 0..3 {
                let dst = frame[idx + c] as f32;
                let src = color[c] as f32;
                frame[idx + c] = (dst + (src - dst) * a).clamp(0.0, 255.0) as u8;
            }
            frame[idx + 3] = 255;
        }
    }
}
