default_prov_config_path <- function() {
  path <- system.file("config", "prov.config.yaml", package = "RExplAnnotator")
  if (nzchar(path)) {
    return(path)
  }

  local_path <- file.path("inst", "config", "prov.config.yaml")
  if (file.exists(local_path)) {
    return(normalizePath(local_path, mustWork = TRUE))
  }

  stop("Default provenance config was not found.", call. = FALSE)
}

load_prov_config <- function(config = NULL, config_path = NULL) {
  if (!is.null(config)) {
    if (!is.list(config)) {
      stop("config must be a list.", call. = FALSE)
    }
    return(config)
  }

  if (is.null(config_path)) {
    config_path <- default_prov_config_path()
  }

  if (!file.exists(config_path)) {
    stop("Config file does not exist: ", config_path, call. = FALSE)
  }

  yaml::read_yaml(config_path)
}

validate_namespaces <- function(ns) {
  if (!is.list(ns) || is.null(names(ns))) {
    stop("Namespaces must be a named list.", call. = FALSE)
  }

  bad_keys <- names(ns)[!nzchar(names(ns))]
  if (length(bad_keys)) {
    stop("All namespace prefixes must have non-empty names.", call. = FALSE)
  }

  bad_vals <- vapply(
    ns,
    function(u) !is.character(u) || length(u) != 1 || !nzchar(u),
    logical(1)
  )

  if (any(bad_vals)) {
    bad <- paste(names(ns)[bad_vals], collapse = ", ")
    stop(
      "All namespace IRIs must be non-empty single strings. Offenders: ",
      bad,
      call. = FALSE
    )
  }

  invisible(TRUE)
}

make_ttl_namespace <- function(yaml_config, g = NULL) {
  prefixes <- yaml_config$ttl$prefixes
  if (is.null(prefixes)) {
    stop("Config must define ttl$prefixes.", call. = FALSE)
  }

  namespaces <- list()
  for (item in prefixes) {
    if (is.null(item$name) || is.null(item$uri)) {
      stop("Each ttl prefix must include name and uri.", call. = FALSE)
    }
    namespaces[[item$name]] <- item$uri
  }

  validate_namespaces(namespaces)
  namespaces
}

.require_rdflib <- function() {
  if (!requireNamespace("rdflib", quietly = TRUE)) {
    stop(
      "The rdflib package is required for RDF graph operations. ",
      "Install or reinstall it with install.packages('rdflib').",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
