get_files_to_check <- function(root, folder = NULL, ignore = character()) {
  files <- list.files(root, full.names = TRUE, recursive = TRUE)
  files <- files[!basename(files) %in% ignore]
  files <- files[grepl("\\.R$", files)]

  package_files <- c("gen_provone.R", "decorator_func.R", "utils.R")
  files <- files[!basename(files) %in% package_files]
  files
}

entity_marking <- function(entity, config) {
  if (is.null(config$program$name)) {
    stop("No program name defined in the config.", call. = FALSE)
  }

  if (!is.character(entity) || length(entity) != 1 || !nzchar(entity)) {
    stop("entity must be a non-empty single string.", call. = FALSE)
  }

  if (grepl(":", entity, fixed = TRUE)) {
    return(entity)
  }

  sprintf("%s:%s", config$program$name, entity)
}

name_concat <- function(...) {
  paste(..., sep = "-")
}

get_unq_id <- function() {
  paste0("id_", format(Sys.time(), "%Y%m%d%H%M%S"), "_", sample(1:1000, 1))
}

get_time_stamp <- function() {
  format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
}

.as_single_string <- function(x, label) {
  if (is.null(x) || length(x) != 1) {
    stop(label, " must be a single value.", call. = FALSE)
  }
  as.character(x)
}
