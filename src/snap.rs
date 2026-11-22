//! Friendlier errors for the confined Snap package.

use std::path::Path;

pub fn is_snap() -> bool {
  std::env::var_os("SNAP").is_some_and(|value| !value.is_empty())
}

pub fn permission_error(directory: &str, symlink: Option<&str>, is_current_directory: bool) -> String {
  let Some(home) = std::env::var("SNAP_REAL_HOME").ok().filter(|home| !home.is_empty()) else {
    return "SNAP_REAL_HOME environment variable is not set. Please, report the issue to the rclip project on \
      GitHub https://github.com/yurijmikhalevich/rclip/issues."
      .into();
  };
  let directory_description = if is_current_directory { "the current directory" } else { directory };
  let path_info = match symlink {
    Some(symlink) if symlink != directory => format!("symlink {symlink} which points to {directory_description}"),
    _ => directory_description.to_string(),
  };
  let is_under =
    |root: &str| directory == root || directory.starts_with(&format!("{root}{}", std::path::MAIN_SEPARATOR));
  if is_under(&home) {
    return format!(
      "rclip doesn't have access to {path_info}. You can resolve this issue by running:\n\n\
      \tsudo snap connect rclip:home\n\n\
      This command will grant rclip the necessary access to the home directory. Afterward, you can try again."
    );
  }
  if is_under("/media") {
    return format!(
      "rclip doesn't have access to {path_info}. You can resolve this issue by running:\n\n\
      \tsudo snap connect rclip:removable-media\n\n\
      This command will grant rclip the necessary access to the \"/media\" directory. Afterward, you can try again."
    );
  }
  "rclip installed with snap cannot access files outside of the home or \"/media\" directories. If you want to use \
  rclip outside of home or \"/media\", file an issue in the rclip project on GitHub \
  https://github.com/yurijmikhalevich/rclip/issues, describe your use case, and consider alternative rclip \
  installation options https://github.com/yurijmikhalevich/rclip#linux."
    .into()
}

/// Exits with an explanation when the snap cannot read `directory`.
pub fn check_permissions(directory: &str, is_current_directory: bool) {
  let denied = matches!(
    std::fs::read_dir(directory).and_then(|mut entries| entries.next().transpose()),
    Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied
  );
  if denied {
    exit_with_permission_error(directory, is_current_directory);
  }
}

/// Exits with an explanation for a directory that the snap could not read while indexing.
pub fn exit_with_permission_error(directory: &str, is_current_directory: bool) -> ! {
  let path = Path::new(directory);
  let (realpath, symlink) = if path.is_symlink() {
    let target = std::fs::canonicalize(path).map_or(directory.to_string(), |target| target.to_string_lossy().into());
    (target, Some(directory))
  } else {
    (directory.to_string(), None)
  };
  eprintln!("{}", permission_error(&realpath, symlink, is_current_directory));
  std::process::exit(1);
}
