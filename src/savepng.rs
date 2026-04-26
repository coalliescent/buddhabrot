//! PNG save with view-parameter metadata.
//!
//! Metadata is written as PNG tEXt chunks. They show up under the "PNG
//! textual" section in ExifTool, ImageMagick `identify -verbose`, and most
//! other image-inspection tools. The values are free-form strings; the
//! rotation matrix is precision-formatted so a view can be reconstructed
//! exactly (16 hex digits per f64 via `{:+e}`).

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::view::View;

pub struct SaveMeta<'a> {
    pub view: &'a View,
    pub palette_name: &'a str,
    pub gamma: f32,
    pub samples: u64,
}

pub fn save_png(rgba: &[u8], w: u32, h: u32, meta: SaveMeta) -> io::Result<PathBuf> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let dir = home.join("buddhabrot");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("buddhabrot-{ts}.png"));
    save_png_to(&path, rgba, w, h, meta)?;
    Ok(path)
}

pub fn save_png_to(
    path: &std::path::Path,
    rgba: &[u8],
    w: u32,
    h: u32,
    meta: SaveMeta,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let v = meta.view;
    let r = &v.rotation;
    // Rotation: four rows, semicolon-separated; each row four whitespace-
    // separated exponent-form floats. Round-trippable to the same f64.
    let rot = format!(
        "{:+e} {:+e} {:+e} {:+e}; {:+e} {:+e} {:+e} {:+e}; \
         {:+e} {:+e} {:+e} {:+e}; {:+e} {:+e} {:+e} {:+e}",
        r[0][0], r[0][1], r[0][2], r[0][3],
        r[1][0], r[1][1], r[1][2], r[1][3],
        r[2][0], r[2][1], r[2][2], r[2][3],
        r[3][0], r[3][1], r[3][2], r[3][3],
    );

    // Swallow tEXt write errors individually — a missing metadata chunk
    // shouldn't prevent the image bytes from being written.
    let _ = encoder.add_text_chunk("Software".into(), "buddhabrot".into());
    let _ = encoder.add_text_chunk(
        "Comment".into(),
        format!(
            "buddhabrot 4D view. axes: X=c_re Y=c_im Z=z_re W=z_im. \
             center=({:+e},{:+e}) half_width={:+e} max_iter={} palette={} gamma={:+e} samples={}",
            v.center[0], v.center[1], v.half_width, v.max_iter,
            meta.palette_name, meta.gamma, meta.samples,
        ),
    );
    let _ = encoder.add_text_chunk("Buddhabrot.Rotation".into(), rot);
    let _ = encoder.add_text_chunk(
        "Buddhabrot.Center".into(),
        format!("{:+e} {:+e}", v.center[0], v.center[1]),
    );
    let _ = encoder.add_text_chunk(
        "Buddhabrot.HalfWidth".into(),
        format!("{:+e}", v.half_width),
    );
    let _ = encoder.add_text_chunk("Buddhabrot.MaxIter".into(), v.max_iter.to_string());
    let _ = encoder.add_text_chunk("Buddhabrot.Palette".into(), meta.palette_name.into());
    let _ = encoder.add_text_chunk("Buddhabrot.Gamma".into(), format!("{:+e}", meta.gamma));
    let _ = encoder.add_text_chunk("Buddhabrot.Samples".into(), meta.samples.to_string());
    let _ = encoder.add_text_chunk(
        "Buddhabrot.Axes".into(),
        "X=c_re Y=c_im Z=z_re W=z_im".into(),
    );
    let _ = encoder.add_text_chunk(
        "Buddhabrot.Dimensions".into(),
        format!("{}x{}", w, h),
    );

    let mut writer = encoder
        .write_header()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    Ok(())
}
