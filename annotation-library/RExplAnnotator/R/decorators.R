find_decorators <- function(file) {
  contents <- readLines(file, warn = FALSE)
  extract_decorators(contents)
}

extract_decorators <- function(lines) {
  pattern <- "(?<=#' @)[A-Za-z0-9_]+\\([^()]*\\)"
  matches <- regmatches(lines, regexpr(pattern, lines, perl = TRUE))
  matches[matches != ""]
}

decorator_to_triples <- function(decorator, config) {
  .process_decorator(decorator, config)
}

process_files <- function(files, config) {
  decorators_list <- list()
  for (file in files) {
    decorators <- find_decorators(file)
    for (decorator in decorators) {
      triples <- decorator_to_triples(decorator, config)
      decorators_list <- c(decorators_list, triples)
    }
  }

  paste(unlist(decorators_list), collapse = "\n")
}

.extract_func_name <- function(s) {
  sub("\\(.*$", "", s)
}

.process_decorator <- function(decorator, config) {
  decorator_type <- .extract_func_name(decorator)
  program_name <- config$program$name

  if (identical(decorator_type, "Func")) {
    return(.process_program_decorator(decorator, program_name, is_function = TRUE))
  }
  if (identical(decorator_type, "Program")) {
    return(.process_program_decorator(decorator, program_name, is_function = FALSE))
  }
  if (identical(decorator_type, "UserInput")) {
    return(.process_user_io_decorator(decorator, program_name, input = TRUE))
  }
  if (identical(decorator_type, "UserOutput")) {
    return(.process_user_io_decorator(decorator, program_name, input = FALSE))
  }
  if (identical(decorator_type, "Channel")) {
    return(.process_channel_decorator(decorator, program_name))
  }

  stop("Unknown decorator type: ", decorator_type, call. = FALSE)
}

.extract_kv_pairs <- function(line) {
  inside <- sub(".*\\((.*)\\).*", "\\1", line)
  pairs <- strsplit(inside, ";\\s*")[[1]]

  kv_list <- lapply(pairs, function(p) {
    parts <- strsplit(p, "->")[[1]]
    if (length(parts) == 2) {
      return(list(key = trimws(parts[1]), value = trimws(parts[2])))
    }
    NULL
  })

  kv_list <- Filter(Negate(is.null), kv_list)
  kv_grouped <- list()
  for (pair in kv_list) {
    kv_grouped[[pair$key]] <- c(kv_grouped[[pair$key]], pair$value)
  }
  kv_grouped
}

.program_name_from_decorator <- function(line, is_function = FALSE) {
  pattern <- if (is_function) {
    "(?<=Func\\()[^\\-\\(\\)]+"
  } else {
    "(?<=Program\\()[^\\-\\(\\)]+"
  }

  pname <- regmatches(line, regexpr(pattern, line, perl = TRUE))
  if (length(pname) == 0 || !nzchar(pname)) {
    stop("Program name not found in decorator.", call. = FALSE)
  }

  gsub("\\s+", "", pname)
}

.process_program_decorator <- function(decorator_line, program_name, is_function = TRUE) {
  object_maps <- list(
    "provone:hasInPort" = "provone:Port",
    "provone:hasOutPort" = "provone:Port"
  )

  kv_list <- .extract_kv_pairs(decorator_line)
  pname <- .program_name_from_decorator(decorator_line, is_function)

  details <- list(
    prog_name = paste0(program_name, ":", pname),
    object_type = kv_list[[pname]][[1]]
  )

  if (is_function) {
    details$tier <- 0
  }

  for (key in names(kv_list)) {
    if (identical(key, pname)) {
      next
    }
    details[[key]] <- kv_list[[key]]
  }

  text <- paste0(details$prog_name, " a ", details$object_type, " .\n")
  for (key in names(details)) {
    if (key %in% c("prog_name", "object_type", "tier", "rdfs:label")) {
      next
    }
    for (value in details[[key]]) {
      text <- paste0(text, details$prog_name, " ", key, " ", details$prog_name, ":", value, " .\n")
    }
  }

  for (key in names(details)) {
    if (!key %in% names(object_maps)) {
      next
    }
    for (value in details[[key]]) {
      text <- paste0(text, details$prog_name, ":", value, " a ", object_maps[[key]], " .\n")
    }
  }

  paste0(text, "\n")
}

.process_channel_decorator <- function(decorator_line, program_name) {
  kv_list <- .extract_kv_pairs(decorator_line)
  details <- list(prog_name = paste0(program_name, ":", kv_list$id[[1]]))

  for (key in names(kv_list)) {
    if (identical(key, "id")) {
      next
    }
    details[[key]] <- kv_list[[key]]
  }

  text <- paste0(details$prog_name, " a provone:Channel .\n")
  for (key in names(details)) {
    if (identical(key, "prog_name")) {
      next
    }
    for (value in details[[key]]) {
      text <- paste0(text, details$prog_name, ":", value, " ", key, " ", details$prog_name, " .\n")
    }
  }

  text
}

.process_user_io_decorator <- function(decorator_line, program_name, input = TRUE) {
  kv_list <- .extract_kv_pairs(decorator_line)
  details <- list(prog_name = paste0(program_name, ":", kv_list$id[[1]]))

  for (key in names(kv_list)) {
    if (identical(key, "id")) {
      next
    }
    details[[key]] <- kv_list[[key]]
  }

  text <- paste0(details$prog_name, "_userport a provone:Port .\n")
  text <- paste0(text, details$prog_name, "_userchannel a provone:Channel .\n")
  text <- paste0(
    text,
    details$prog_name,
    "_userport provone:connectsTo ",
    details$prog_name,
    "_userchannel .\n"
  )

  for (key in names(details)) {
    if (identical(key, "prog_name")) {
      next
    }
    for (value in details[[key]]) {
      text <- paste0(text, details$prog_name, ":", value, " ", key, " ", details$prog_name, "_userchannel .\n")
    }
  }

  text
}
