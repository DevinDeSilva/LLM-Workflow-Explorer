provProgram <- function(
  config,
  name,
  hasInPort,
  hasOutPort,
  hasSubProgram = NULL,
  metadata = list(),
  isAITask = FALSE,
  aiTask = list(),
  ...
) {
  if (is.null(config$program$name)) {
    stop("No program name defined in the config.", call. = FALSE)
  }

  prog_name <- entity_marking(name, config)
  details <- list(
    name = prog_name,
    hasInPort = hasInPort,
    hasOutPort = hasOutPort
  )

  config$add_to_graph(prog_name, "a", "provone:Program")

  for (port_key in names(hasInPort)) {
    port <- hasInPort[[port_key]]
    port_ident <- name_concat(prog_name, port$name)

    config$add_to_graph(prog_name, "provone:hasInPort", port_ident)
    config$add_to_graph(port_ident, "a", "provone:Port")

    if (!is.null(port$default)) {
      stop("provone:hasDefault is not implemented yet.", call. = FALSE)
    }

    add_metadata_to_object(port_ident, port$metadata %||% list(), config)

    details$hasInPort[[port_key]]$name <- port_ident
    details$hasInPort[[port_key]]$port_key <- port$name
  }

  for (port_key in names(hasOutPort)) {
    port <- hasOutPort[[port_key]]
    port_ident <- name_concat(prog_name, port$name)

    config$add_to_graph(prog_name, "provone:hasOutPort", port_ident)
    config$add_to_graph(port_ident, "a", "provone:Port")

    if (!is.null(port$default)) {
      stop("provone:hasDefault is not implemented yet.", call. = FALSE)
    }

    add_metadata_to_object(port_ident, port$metadata %||% list(), config)

    details$hasOutPort[[port_key]]$name <- port_ident
    details$hasOutPort[[port_key]]$port_key <- port$name
  }

  if (!is.null(hasSubProgram)) {
    for (sub_program in hasSubProgram) {
      config$add_to_graph(prog_name, "provone:hasSubProgram", sub_program$name)
    }
    details$hasSubProgram <- hasSubProgram
  }

  if (!is.null(metadata)) {
    add_metadata_to_object(prog_name, metadata, config)
  }

  if (isAITask && !is.null(aiTask)) {
    if (!is.list(aiTask)) {
      stop("AI tasks must be a list.", call. = FALSE)
    }

    ai_task_name <- entity_marking(name_concat("Generative_Task", name), config)
    config$add_to_graph(ai_task_name, "a", "workflow:Generative_Task")

    ai_method_name <- entity_marking(name_concat("LLM", name), config)
    config$add_to_graph(ai_method_name, "a", "workflow:Large_Language_Models")

    config$add_to_graph(ai_task_name, "prov:used", ai_method_name)
    config$add_to_graph(
      ai_method_name,
      "workflow:llm_model",
      aiTask$llm_model %||% "",
      literal = TRUE,
      lang = "en",
      dtype = "xsd:string"
    )

    ai_task_input <- list()
    for (inp in names(aiTask$input)) {
      inp_name <- entity_marking(name_concat("LLM", name, "Input", inp), config)

      config$add_to_graph(inp_name, "a", "provone:Data")
      config$add_to_graph(ai_method_name, "sio:SIO_000230", inp_name)
      config$add_to_graph(
        inp_name,
        "prov:value",
        aiTask$input[[inp]],
        literal = TRUE,
        lang = "en",
        dtype = "xsd:string"
      )

      ai_task_input[[inp]] <- list(
        name = inp_name,
        value = aiTask$input[[inp]],
        metadata = list()
      )
    }

    ai_output_name <- entity_marking(name_concat("LLM_Output", name), config)
    config$add_to_graph(ai_output_name, "a", "workflow:Large_Language_Model_Output")
    config$add_to_graph(ai_method_name, "sio:SIO_000229", ai_output_name)
    config$add_to_graph(ai_output_name, "sio:SIO_000232", ai_method_name)
    config$add_to_graph(ai_output_name, "sio:SIO_000202", prog_name)

    if (!is.null(aiTask$metadata)) {
      add_metadata_to_object(ai_method_name, aiTask$metadata, config)
      add_metadata_to_object(ai_task_name, aiTask$metadata, config)
    }

    details$aiTask <- aiTask
  }

  details
}

provChannel <- function(config, name, connectsTo, metadata = list(), ...) {
  if (is.null(config$program$name)) {
    stop("No program name defined in the config.", call. = FALSE)
  }

  prog_name <- entity_marking(name, config)
  config$add_to_graph(prog_name, "a", "provone:Channel")

  if (!is.null(connectsTo)) {
    for (port in connectsTo) {
      config$add_to_graph(port$name, "provone:connectsTo", prog_name)
    }
  }

  if (!is.null(metadata)) {
    add_metadata_to_object(prog_name, metadata, config)
  }

  list(name = prog_name, connectsTo = connectsTo)
}

.process_exe_literal <- function(flow_data, flow, prog, execution_name, config, direction) {
  data_id <- get_unq_id()
  data_name <- entity_marking(name_concat("Data", data_id, flow), config)

  if (identical(direction, "input")) {
    usage_name <- entity_marking(name_concat("Usage", data_id, flow), config)

    config$add_to_graph(data_name, "a", "provone:Data")
    config$add_to_graph(data_name, "rdfs:label", flow, literal = TRUE, lang = "en", dtype = "xsd:string")
    config$add_to_graph(data_name, "prov:value", flow_data[[flow]]$value, literal = TRUE, lang = "en", dtype = "xsd:string")

    config$add_to_graph(usage_name, "a", "prov:Usage")
    config$add_to_graph(usage_name, "provone:hadInPort", prog$hasInPort[[flow]]$name)
    config$add_to_graph(usage_name, "provone:hadEntity", data_name)

    config$add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
    config$add_to_graph(execution_name, "prov:used", data_name)
  } else {
    generation_name <- entity_marking(name_concat("Generation", data_id, flow), config)

    config$add_to_graph(data_name, "a", "provone:Data")
    config$add_to_graph(data_name, "rdfs:label", flow, literal = TRUE, lang = "en", dtype = "xsd:string")
    config$add_to_graph(data_name, "prov:value", flow_data[[flow]]$value, literal = TRUE, lang = "en", dtype = "xsd:string")

    config$add_to_graph(generation_name, "a", "prov:Generation")
    config$add_to_graph(generation_name, "provone:hadOutPort", prog$hasOutPort[[flow]]$name)
    config$add_to_graph(generation_name, "provone:hadEntity", data_name)

    config$add_to_graph(execution_name, "prov:qualifiedGeneration", generation_name)
    config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)
  }

  list(id = data_id, name = data_name)
}

.collection_entry <- function(flow_data, flow, prog, execution_name, config, direction) {
  data_id <- get_unq_id()
  data_name <- entity_marking(name_concat("Collection", data_id, flow), config)
  components <- list()

  if (identical(direction, "input")) {
    usage_name <- entity_marking(name_concat("Usage", data_id, flow), config)

    config$add_to_graph(data_name, "a", "provone:Collection")
    config$add_to_graph(usage_name, "a", "prov:Usage")
    config$add_to_graph(usage_name, "provone:hadInPort", prog$hasInPort[[flow]]$name)

    config$add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
    config$add_to_graph(execution_name, "prov:used", data_name)

    components$usage_name <- usage_name
  } else {
    generation_name <- entity_marking(name_concat("Generation", data_id, flow), config)

    config$add_to_graph(data_name, "a", "provone:Collection")
    config$add_to_graph(generation_name, "a", "prov:Generation")
    config$add_to_graph(generation_name, "provone:hadOutPort", prog$hasOutPort[[flow]]$name)

    config$add_to_graph(execution_name, "prov:qualifiedGeneration", generation_name)
    config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)

    components$generation_name <- generation_name
  }

  list(id = data_id, name = data_name, components = components)
}

.process_exe_list <- function(flow_data, flow, prog, execution_name, config, direction) {
  collection_name <- .collection_entry(flow_data, flow, prog, execution_name, config, direction)

  data_names_list <- list()
  for (d in flow_data[[flow]]$value) {
    data_id <- get_unq_id()
    data_name <- entity_marking(name_concat("Data", data_id, flow), config)

    config$add_to_graph(data_name, "a", "provone:Data")
    config$add_to_graph(data_name, "rdfs:label", flow, literal = TRUE, lang = "en", dtype = "xsd:string")
    config$add_to_graph(data_name, "prov:value", d, literal = TRUE, lang = "en", dtype = "xsd:string")
    config$add_to_graph(collection_name$name, "provone:hadMember", data_name)
    config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)

    if (identical(direction, "input")) {
      config$add_to_graph(collection_name$components$usage_name, "provone:hadEntity", data_name)
    } else {
      config$add_to_graph(collection_name$components$generation_name, "provone:hadEntity", data_name)
    }

    data_names_list <- append(data_names_list, list(list(id = data_id, name = data_name)))
  }

  list(collection = collection_name, members = data_names_list)
}

.process_exe_prov_data <- function(flow_data, flow, prog, execution_name, config, direction, semantic_map = NULL) {
  flow_id <- get_unq_id()
  data_name <- flow_data[[flow]]$value$name

  if (identical(direction, "input")) {
    usage_name <- entity_marking(name_concat("Usage", flow_id, flow), config)

    config$add_to_graph(usage_name, "a", "prov:Usage")
    config$add_to_graph(usage_name, "provone:hadInPort", prog$hasInPort[[flow]]$name)
    config$add_to_graph(usage_name, "provone:hadEntity", data_name)

    config$add_to_graph(execution_name, "prov:qualifiedUsage", usage_name)
    config$add_to_graph(execution_name, "prov:used", data_name)
    config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)
  } else {
    stop("Output data is not supported for data_type = 'prov-data'.", call. = FALSE)
  }

  flow_data[[flow]]$value
}

.process_exe_df <- function(flow_data, flow, prog, execution_name, config, direction, semantic_map = NULL) {
  if (!is.data.frame(flow_data[[flow]]$value)) {
    stop("Input data is not a data frame.", call. = FALSE)
  }

  collection_name <- .collection_entry(flow_data, flow, prog, execution_name, config, direction)
  sel_df <- flow_data[[flow]]$value

  if (is.null(semantic_map)) {
    semantic_map <- list()
    for (col in names(sel_df)) {
      semantic_map[[col]] <- sprintf("DFColumn:%s", col)
    }
  }

  data_names_list <- list()
  for (i in seq_len(nrow(sel_df))) {
    data_id <- get_unq_id()
    data_name <- entity_marking(name_concat("Data", data_id, flow), config)

    config$add_to_graph(data_name, "a", "provone:Data")
    config$add_to_graph(data_name, "rdfs:label", paste0("row_", i), literal = TRUE, lang = "en", dtype = "xsd:string")
    config$add_to_graph(collection_name$name, "provone:hadMember", data_name)
    config$add_to_graph(execution_name, "prov:used", data_name)
    config$add_to_graph(data_name, "prov:wasGeneratedBy", execution_name)

    if (identical(direction, "input")) {
      config$add_to_graph(collection_name$components$usage_name, "provone:hadEntity", data_name)
    } else {
      config$add_to_graph(collection_name$components$generation_name, "provone:hadEntity", data_name)
    }

    for (col in names(sel_df)) {
      pred <- semantic_map[[col]] %||% sprintf("DFColumn:%s", col)
      value <- as.character(sel_df[i, col][[1]])
      config$add_to_graph(data_name, pred, value, literal = TRUE, lang = "en", dtype = "xsd:string")
    }

    data_names_list <- append(data_names_list, list(list(id = data_id, name = data_name)))
  }

  list(collection = collection_name, members = data_names_list)
}

provMakeList <- function(config, entities, metadata = list(), ...) {
  collection_id <- get_unq_id()
  collection_name <- entity_marking(collection_id, config)

  config$add_to_graph(collection_name, "a", "provone:Collection")
  for (ent in entities) {
    config$add_to_graph(collection_name, "provone:hadMember", ent)
  }

  list(
    collection = list(name = collection_name, id = collection_id),
    members = entities
  )
}

provProgramExecution <- function(
  config,
  prog,
  inputs,
  outputs,
  user,
  semantic_map = NULL,
  usesAI = FALSE,
  usedAIInfo = list(),
  metadata = list(),
  ...
) {
  if (is.null(prog$name)) {
    stop("No program name defined in prog.", call. = FALSE)
  }

  execution_id <- get_unq_id()
  execution_name <- entity_marking(execution_id, config)
  user_name <- entity_marking(user, config)
  association_name <- entity_marking(name_concat(prog$name, "Association", execution_id), config)

  config$add_to_graph(execution_name, "a", "provone:Execution")
  config$add_to_graph(user_name, "a", "prov:Agent")
  config$add_to_graph(association_name, "a", "prov:Association")

  config$add_to_graph(execution_name, "prov:wasAssociatedWith", user_name)
  config$add_to_graph(association_name, "prov:hadPlan", prog$name)
  config$add_to_graph(association_name, "prov:agent", user_name)
  config$add_to_graph(execution_name, "prov:qualifiedAssociation", association_name)

  recordsInputs <- list()
  for (inp in names(inputs)) {
    data_type <- inputs[[inp]]$data_type
    if (identical(data_type, "literal")) {
      data_name <- .process_exe_literal(inputs, inp, prog, execution_name, config, "input")
    } else if (identical(data_type, "prov-data")) {
      data_name <- .process_exe_prov_data(inputs, inp, prog, execution_name, config, "input")
    } else if (identical(data_type, "data_frame")) {
      data_name <- .process_exe_df(inputs, inp, prog, execution_name, config, "input", semantic_map)
    } else if (identical(data_type, "list")) {
      data_name <- .process_exe_list(inputs, inp, prog, execution_name, config, "input")
    } else {
      stop("Unsupported input data type: ", data_type, call. = FALSE)
    }

    recordsInputs[[inp]] <- data_name
  }

  recordsOutputs <- list()
  for (out in names(outputs)) {
    data_type <- outputs[[out]]$data_type
    if (identical(data_type, "literal")) {
      data_name <- .process_exe_literal(outputs, out, prog, execution_name, config, "output")
    } else if (identical(data_type, "data_frame")) {
      data_name <- .process_exe_df(outputs, out, prog, execution_name, config, "output", semantic_map)
    } else if (identical(data_type, "list")) {
      data_name <- .process_exe_list(outputs, out, prog, execution_name, config, "output")
    } else {
      stop("Unsupported output data type: ", data_type, call. = FALSE)
    }

    recordsOutputs[[out]] <- data_name
  }

  if (usesAI && !is.null(usedAIInfo)) {
    if (!is.list(usedAIInfo)) {
      stop("AI tasks must be a list.", call. = FALSE)
    }

    ai_task_name <- entity_marking(name_concat("Generative_Task", execution_id), config)
    config$add_to_graph(ai_task_name, "a", "workflow:Generative_Task")

    ai_method_name <- entity_marking(name_concat("LLM", execution_id), config)
    config$add_to_graph(ai_method_name, "a", "workflow:Large_Language_Models")

    config$add_to_graph(ai_task_name, "prov:used", ai_method_name)
    config$add_to_graph(
      ai_method_name,
      "workflow:llm_model",
      usedAIInfo$llm_model %||% "",
      literal = TRUE,
      lang = "en",
      dtype = "xsd:string"
    )

    config$add_to_graph(ai_task_name, "sio:SIO_000313", execution_name)
    config$add_to_graph(execution_name, "sio:SIO_000369", ai_task_name)

    for (inp in names(usedAIInfo$input)) {
      data_type <- inputs[[inp]]$data_type
      if (identical(data_type, "literal") || identical(data_type, "prov-data")) {
        config$add_to_graph(ai_method_name, "sio:SIO_000230", recordsInputs[[inp]]$name)
      } else if (identical(data_type, "data_frame") || identical(data_type, "list")) {
        for (mem in recordsInputs[[inp]]$members) {
          config$add_to_graph(ai_method_name, "sio:SIO_000230", mem$name)
        }
      } else {
        stop("Unsupported input data type: ", data_type, call. = FALSE)
      }
    }

    ai_output_name <- entity_marking(name_concat("LLM_Output", execution_id), config)
    config$add_to_graph(ai_output_name, "a", "workflow:Large_Language_Model_Output")
    config$add_to_graph(ai_method_name, "sio:SIO_000229", ai_output_name)
    config$add_to_graph(ai_output_name, "sio:SIO_000232", ai_method_name)

    for (out in names(usedAIInfo$output)) {
      data_type <- outputs[[out]]$data_type
      if (identical(data_type, "literal")) {
        config$add_to_graph(ai_output_name, "sio:SIO_000202", recordsOutputs[[out]]$name)
      } else if (identical(data_type, "data_frame") || identical(data_type, "list")) {
        for (mem in recordsOutputs[[out]]$members) {
          config$add_to_graph(ai_output_name, "sio:SIO_000202", mem$name)
        }
      } else {
        stop("Unsupported output data type: ", data_type, call. = FALSE)
      }
    }

    if (!is.null(usedAIInfo$metadata)) {
      add_metadata_to_object(ai_method_name, usedAIInfo$metadata, config)
      add_metadata_to_object(ai_task_name, usedAIInfo$metadata, config)
    }
  }

  if (!is.null(metadata)) {
    add_metadata_to_object(execution_name, metadata, config)
  }

  list(
    name = execution_name,
    inputs = recordsInputs,
    outputs = recordsOutputs,
    user = user_name
  )
}

provWasInformedBy <- function(config, informed, informing, ...) {
  if (is.null(informed$name) || is.null(informing$name)) {
    stop("Both informed and informing must have a name defined.", call. = FALSE)
  }

  config$add_to_graph(informed$name, "prov:wasInformedBy", informing$name)
  invisible(TRUE)
}

provWasPartOf <- function(config, informed, informing, ...) {
  if (is.null(informed$name) || is.null(informing$name)) {
    stop("Both informed and informing must have a name defined.", call. = FALSE)
  }

  config$add_to_graph(informed$name, "provone:wasPartOf", informing$name)
  invisible(TRUE)
}

create_prov_module <- function(config = NULL, config_path = NULL) {
  .require_rdflib()

  env <- new.env(parent = emptyenv())
  env$graph <- rdflib::rdf()
  env$config <- load_prov_config(config = config, config_path = config_path)
  env$config$namespaces <- make_ttl_namespace(env$config, env$graph)

  program_prefix <- env$config$program$name
  if (!program_prefix %in% names(env$config$namespaces)) {
    stop(
      "The program name '",
      program_prefix,
      "' must also be defined as a ttl prefix.",
      call. = FALSE
    )
  }

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

  env$Programs <- list()
  env$provProgram <- function(...) provProgram(config = env$config, ...)
  env$provChannel <- function(...) provChannel(config = env$config, ...)
  env$provProgramExecution <- function(...) provProgramExecution(config = env$config, ...)
  env$provWasInformedBy <- function(...) provWasInformedBy(config = env$config, ...)
  env$provWasPartOf <- function(...) provWasPartOf(config = env$config, ...)
  env$provMakeList <- function(...) provMakeList(config = env$config, ...)
  env$add_to_namespace <- function(prefix, uri) add_to_namespace(prefix, uri, env = env)
  env$save_prov_graph <- function() save_prov_graph(.graph = env$graph, config = env$config)

  env
}

prov_module <- create_prov_module
