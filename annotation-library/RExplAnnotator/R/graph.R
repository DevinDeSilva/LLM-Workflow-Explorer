curie <- function(x, ns, default_prefix = NULL, allow_bare = FALSE) {
  if (!is.character(x) || length(x) != 1 || !nzchar(x)) {
    stop("x must be a non-empty single string.", call. = FALSE)
  }

  if (is.list(ns)) {
    ns <- unlist(ns, use.names = TRUE)
  }
  if (!is.character(ns) || is.null(names(ns))) {
    stop("ns must be a named character vector or named list.", call. = FALSE)
  }

  if (identical(x, "a")) {
    return("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
  }

  if (grepl("^(https?|urn):", x, ignore.case = TRUE)) {
    return(x)
  }

  if (grepl(":", x, fixed = TRUE)) {
    parts <- strsplit(x, ":", fixed = TRUE)[[1]]
    prefix <- parts[1]
    local <- paste(parts[-1], collapse = ":")
    if (!nzchar(local)) {
      stop("Empty local part in CURIE: ", x, call. = FALSE)
    }
    if (!prefix %in% names(ns)) {
      stop("Unknown prefix in CURIE: ", x, call. = FALSE)
    }
    return(paste0(ns[[prefix]], local))
  }

  if (!is.null(default_prefix)) {
    if (!default_prefix %in% names(ns)) {
      stop("default_prefix '", default_prefix, "' not found in ns.", call. = FALSE)
    }
    return(paste0(ns[[default_prefix]], x))
  }

  if (allow_bare) {
    return(x)
  }

  stop("Not a CURIE and not a full IRI: ", x, call. = FALSE)
}

add_to_graph <- function(
  s,
  p,
  o,
  .g,
  namespaces,
  literal = FALSE,
  lang = NULL,
  dtype = NULL
) {
  .require_rdflib()

  s <- curie(s, namespaces)
  p <- curie(p, namespaces)

  if (literal) {
    o <- as.character(o)
    if (!is.null(lang)) {
      o <- paste0(o, "@", lang)
    }
    if (!is.null(dtype)) {
      dtype <- curie(dtype, namespaces)
      o <- paste0(o, "^^<", dtype, ">")
    }
    rdflib::rdf_add(.g, s, p, o, objectType = "literal")
  } else {
    o <- curie(o, namespaces)
    rdflib::rdf_add(.g, s, p, o)
  }

  invisible(.g)
}

add_metadata_to_object <- function(object_name, metadata, config) {
  if (!is.list(metadata)) {
    stop("Metadata must be a list.", call. = FALSE)
  }

  for (n in names(metadata)) {
    value <- metadata[[n]]
    if (!is.character(value) || length(value) != 1) {
      stop("Metadata values must be single strings.", call. = FALSE)
    }
    config$add_to_graph(
      object_name,
      n,
      value,
      literal = TRUE,
      lang = "en",
      dtype = "xsd:string"
    )
  }

  invisible(TRUE)
}

save_prov_graph <- function(.graph, config) {
  .require_rdflib()

  metadata <- jsonlite::toJSON(
    list(
      generatedBy = config$program$name,
      generatedAt = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      namespaces = config$namespaces
    ),
    pretty = TRUE,
    auto_unbox = TRUE
  )

  writeLines(metadata, con = config$ttl$metadata_path)

  rdflib::rdf_serialize(
    .graph,
    config$ttl$save_path,
    format = config$ttl$format %||% "turtle",
    namespace = config$namespaces,
    prefix = names(config$namespaces)
  )

  invisible(config$ttl$save_path)
}

add_to_namespace <- function(prefix, uri, env) {
  if (prefix %in% names(env$config$namespaces)) {
    return(invisible(TRUE))
  }

  env$config$namespaces[[prefix]] <- uri
  validate_namespaces(env$config$namespaces)
  env$config$add_to_graph <- function(s, p, o, literal = FALSE, lang = NULL, dtype = NULL) {
    add_to_graph(
      s,
      p,
      o,
      .g = env$graph,
      namespaces = env$config$namespaces,
      literal = literal,
      lang = lang,
      dtype = dtype
    )
  }
  invisible(TRUE)
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
