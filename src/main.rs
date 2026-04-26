mod app;
mod cli;
mod composer;
mod gizmo;
mod input;
mod metadata;
mod onion;
mod orbit;
mod overlay;
mod palette;
mod render;
mod rotation_interp;
mod sampler;
mod savepng;
mod session;
mod timeline_save;
mod videorender;
mod view;
mod zoompan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("render-frame") => cli::render_frame(&args[2..]),
        Some("render-video") => cli::render_video(&args[2..]),
        Some("help") | Some("--help") | Some("-h") => {
            cli::print_help();
            Ok(())
        }
        _ => app::run(),
    }
}
