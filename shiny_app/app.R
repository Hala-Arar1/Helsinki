library(shiny)       # for building the interactive web app
library(DT)          # for rendering interactive data tables
library(dplyr)       # for data manipulation
library(readr)       # for reading CSV files
library(shinyjs)     # for JavaScript integration (e.g., toggle dark mode)
library(htmltools)   # for safe HTML rendering

# Load CSV Data
# Read preprocessed CSV file with PubMed preview data
preview_df <- read.csv("data/pubmed_preview_20.csv", stringsAsFactors = FALSE)

# Placeholder
# Ensure required columns exist; fill missing ones with NA
required_cols <- c(
  # Unique ID (combination of protein + paper)
  "Protein-Paper ID",
  
  # Protein metadata
  "AC",
  "Protein Name",
  "Gene Name",
  "OS",
  
  # Publication metadata
  "PMID",
  "Title",
  "Abstract",
  "Journal",
  "Authors",
  "Date Published",
  "Source",
  
  # Label info
  "Autoregulatory Type",
  "Polarity"
)

for (col in required_cols) {
  if (!(col %in% colnames(preview_df))) {
    preview_df[[col]] <- NA
  }
}

# Rename publication date column
if ("PubDate" %in% colnames(preview_df)) {
  preview_df$`Date Published` <- preview_df$PubDate
}

# Mock values for UI control fields
# Fill in missing values with mock or default data to avoid UI errors
set.seed(123)  # ensure reproducibility
if (!("Protein-Paper ID" %in% names(preview_df)) || all(is.na(preview_df$`Protein-Paper ID`))) {
  preview_df$`Protein-Paper ID` <- paste(preview_df$`Protein Name`, preview_df$PMID, sep = "_")
}

if (all(is.na(preview_df$AC))) {
  preview_df$AC <- paste0("AC", sprintf("%04d", seq_len(nrow(preview_df))))
}

if (all(is.na(preview_df$`Protein Name`))) {
  preview_df$`Protein Name` <- paste("Protein", seq_len(nrow(preview_df)))
}

if (all(is.na(preview_df$`Gene Name`))) {
  preview_df$`Gene Name` <- paste("Gene", seq_len(nrow(preview_df)))
}

if (all(is.na(preview_df$OS))) {
  preview_df$OS <- sample(c("Homo sapiens", "Escherichia coli", "Saccharomyces cerevisiae", "Arabidopsis thaliana"), 
                          nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$Journal))) {
  preview_df$Journal <- sample(c("Nature", "Science", "Cell Reports", "J. Bacteriol"), nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$Authors))) {
  preview_df$Authors <- replicate(
    nrow(preview_df),
    paste(sample(c("Smith", "Johnson", "Lee", "Patel", "Zhang", "Garcia", "Nguyen", "Kumar"), 3, replace = FALSE),
          collapse = ", ")
  )
}

if (all(is.na(preview_df$Abstract))) {
  # Generate placeholder abstracts if all are missing
  preview_df$Abstract <- replicate(nrow(preview_df), paste(
    "Autoregulatory mechanisms play a pivotal role in maintaining protein homeostasis within the cell.",
    "In this study, we examine a broad range of autoregulatory behaviors across multiple protein families,",
    "highlighting the interplay between transcriptional feedback loops and post-translational modifications.",
    "Using high-throughput screening methods and sequence alignment analyses,",
    "we identify conserved domains responsible for self-activation and inhibition.",
    "Experimental results reveal that specific mutations can disrupt autoregulatory balance,",
    "leading to aberrant expression profiles and potential pathophysiological consequences.",
    "Furthermore, our data suggest that environmental stimuli such as oxidative stress or nutrient deprivation",
    "can dynamically modulate autoregulatory responses. These findings contribute to a deeper understanding",
    "of autoregulatory logic and provide potential targets for therapeutic intervention in diseases",
    "where self-regulation of proteins is impaired."
  ))
}

if (all(is.na(preview_df$`Autoregulatory Type`))) {
  preview_df$`Autoregulatory Type` <- sample(c("autoregulation", "autoinhibition", "autolysis"), nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$Polarity))) {
  preview_df$Polarity <- sample(c("positive", "neutral", "negative"), nrow(preview_df), replace = TRUE)
}

if (!("Source" %in% names(preview_df)) || all(is.na(preview_df$Source))) {
  preview_df$Source <- sample(
    c("UniProt", "Other DB", "Manual Upload"),
    nrow(preview_df),
    replace = TRUE
  )
}

# Final data frame to be used
# Subset and format data to match UI expectations
df <- preview_df[, required_cols]

# Clean and Standardize Dates
# Fix some missing value in Date Published
df$`Date Published` <- preview_df$`Date Published`

# Case 1: "2020-05-NA" → "2020-05-01"
df$`Date Published` <- gsub("-NA$", "-01", df$`Date Published`)

# Case 2: "2020-NA-NA" → "2020-01-01"
df$`Date Published` <- gsub("-NA-NA", "-01-01", df$`Date Published`)

# Case 3: "2020" → "2020-01-01"
df$`Date Published` <- ifelse(grepl("^\\d{4}$", df$`Date Published`),
                              paste0(df$`Date Published`, "-01-01"),
                              df$`Date Published`)

# Transfer Date Type
df$`Date Published` <- as.Date(df$`Date Published`, format = "%Y-%m-%d")


# Define UI
# Header UI reused across all tabs
header_ui <- div(
  class = "header-section",
  div(class = "app-title", img(src = "logo.png"), "SOORENA"),
  div(class = "header-logos",
      img(src = "logoHelsinki.png"),
      img(src = "logoUBC.png"),
      img(src = "logoJafariLab.png")
  )
)

# Define UI
ui <- navbarPage(
  title = NULL,
  id = "main_nav",
  
  # Tab: Search and Main App Interface
  tabPanel(
    title = "Search",
    fluidPage(
      useShinyjs(),
      tags$head(
        tags$style(HTML("
          body {
            background-color: #f9f9f9;
          }
          .header-section {
            background-image: url('header_img.png');
            background-size: 50% auto;
            background-position: center;
            background-repeat: no-repeat;
            background-color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 20px;
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
          }
          .app-title {
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
            display: flex;
            align-items: center;
          }
          .app-title img {
            height: 120px;
            margin-right: 20px;
          }
          .header-logos {
            display: flex;
            gap: 20px;
          }
          .header-logos img {
            height: 120px;
          }
          .filter-panel {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
          }
          .btn-warning {
            margin-top: 10px;
          }
          .dataTables_wrapper {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 0 30px 20px 30px;
          }
        "))
      ),
      
      # Header section
      header_ui,
      
      # Search and Filter Controls
      div(class = "filter-panel",
          fluidRow(
            column(
              width = 8,
              fluidRow(
                column(3, textInput("ac", "AC", placeholder = "Search AC...")),
                column(3, textInput("protein_name", "Protein Name", placeholder = "Search protein...")),
                column(3, textInput("gene_name", "Gene Name", placeholder = "Search gene...")),
                column(3, selectInput("os", "OS", choices = c("All", sort(unique(na.omit(df$OS)))), multiple = TRUE, selected = "All"))
              ),
              fluidRow(
                column(3, textInput("pmid", "PMID", placeholder = "Search PMID...")),
                column(3, textInput("author", "Author", placeholder = "Search author...")),
                column(3, selectInput("journal", "Journal", choices = c("All", sort(unique(na.omit(df$Journal)))), multiple = TRUE, selected = "All")),
                column(3, dateRangeInput("date_range", "Date Published",
                                         start = min(df$`Date Published`, na.rm = TRUE),
                                         end = max(df$`Date Published`, na.rm = TRUE)))
              ),
              fluidRow(
                column(3, selectInput("type", "Autoregulatory Type", choices = c("All", sort(unique(na.omit(df$`Autoregulatory Type`)))), multiple = TRUE, selected = "All")),
                column(3, selectInput("polarity", "Polarity", choices = c("All", sort(unique(na.omit(df$Polarity)))), multiple = TRUE, selected = "All"))
              )
            ),
            column(
              width = 4,
              textAreaInput("search", "Search Title / Abstract", placeholder = "Type or paste any text...", height = "120px"),
              actionButton("reset_filters", "Reset Filters", class = "btn-warning")
            )
          )
      ),
      
      # Download Button
      div(style = "margin: 0 30px;",
          downloadButton("download_csv", "Download CSV", class = "btn-primary mb-3")
      ),
      
      # Display Table
      div(style = "margin: 0 30px;",
          DTOutput("result_table"))
    )
  ),
  
  # Tab: Statistics
  tabPanel(
    title = "Statistics",
    fluidPage(
      header_ui,
      h2("Statistics"),
    )
  ),
  
  # Tab: Patch Notes
  tabPanel(
    title = "Patch Notes",
    fluidPage(
      header_ui,
      h2("Patch Notes"),
      DT::dataTableOutput("patch_notes_table")
    )
  ),
  
  # Tab: About Us
  tabPanel(
    title = "About Us",
    fluidPage(
      header_ui,
      h2("Project Contributors"),
      tags$ul(
        tags$li("Alexandra Zhou – University of British Columbia"),
        tags$li("Hala Arar – University of British Columbia"),
        tags$li("Mingyang Zhang – University of British Columbia"),
        tags$li("Zheng He – University of British Columbia"),
      ),
      h2("Mentor & Partner"),
      tags$ul(
        tags$li("Mohieddin Jafari – University of Helsinki (Partner)"), 
        tags$li("Payman Nickchi – University of British Columbia (Mentor)")
      )
    )
  ),
)


# Define Server Logic
server <- function(input, output, session) {
  # Save default date range for reset
  default_date_range <- range(df$`Date Published`, na.rm = TRUE)
  
  # Toggle Dark Mode
  observeEvent(input$toggle_dark, {
    runjs("document.body.classList.toggle('dark-mode');")
  })
  
  # Download csv button
  output$download_csv <- downloadHandler(
    filename = function() {
      paste0("filtered_results_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(filtered_data(), file, row.names = FALSE)
    }
  )
  
  # Show Full text
  observeEvent(input$show_full_text, {
    showModal(modalDialog(
      title = paste("Full", input$show_full_text$field),
      div(style = "white-space: pre-wrap; font-family: sans-serif;", input$show_full_text$text),
      easyClose = TRUE,
      footer = modalButton("Close"),
      size = "m"
    ))
  })
  
  # Reset all filters to default state
  observeEvent(input$reset_filters, {
    updateTextInput(session, "ac", value = "")
    updateTextInput(session, "protein_name", value = "")
    updateTextInput(session, "gene_name", value = "")
    updateTextInput(session, "pmid", value = "")
    updateTextInput(session, "author", value = "")
    updateSelectInput(session, "journal", selected = "All")
    updateSelectInput(session, "os", selected = "All")
    updateSelectInput(session, "type", selected = "All")
    updateSelectInput(session, "polarity", selected = "All")
    updateDateRangeInput(session, "date_range",
                         start = default_date_range[1],
                         end = default_date_range[2])
    updateTextAreaInput(session, "search", value = "")
  })
  
  # Filtering Logic
  filtered_data <- reactive({
    result <- df
    print(paste("Initial rows:", nrow(result)))
    
    # Journal filter
    if (!is.null(input$journal) && !"All" %in% input$journal && length(input$journal) > 0) {
      result <- result %>% filter(Journal %in% input$journal)
    }
    print(paste("Rows after Journal filter:", nrow(result)))
    
    # Type filter
    if (!is.null(input$type) && !"All" %in% input$type && length(input$type) > 0) {
      result <- result %>% filter(`Autoregulatory Type` %in% input$type)
    }
    print(paste("Rows after Type filter:", nrow(result)))
    
    # Polarity filter
    if (!is.null(input$polarity) && !"All" %in% input$polarity && length(input$polarity) > 0) {
      result <- result %>% filter(Polarity %in% input$polarity)
    }
    print(paste("Rows after Polarity filter:", nrow(result)))
    
    # OS filter
    if (!is.null(input$os) && !"All" %in% input$os && length(input$os) > 0) {
      result <- result %>% filter(OS %in% input$os)
    }
    print(paste("Rows after OS filter:", nrow(result)))
    
    # Date range filter
    if (!is.null(input$date_range)) {
      start_date <- input$date_range[1]
      end_date <- input$date_range[2]
      if (!is.na(start_date)) {
        result <- result %>% filter(`Date Published` >= start_date)
      }
      if (!is.na(end_date)) {
        result <- result %>% filter(`Date Published` <= end_date)
      }
    }
    print(paste("Rows after Date filter:", nrow(result)))
    
    # AC search
    if (!is.null(input$ac) && nzchar(input$ac)) {
      terms <- trimws(unlist(strsplit(input$ac, ",")))
      pattern <- paste0("\\b(", paste(terms, collapse = "|"), ")\\b")
      result <- result %>% filter(grepl(pattern, AC, ignore.case = TRUE))
    }
    print(paste("Rows after AC search:", nrow(result)))
    
    # Protein Name search
    if (!is.null(input$protein_name) && nzchar(input$protein_name)) {
      terms <- trimws(unlist(strsplit(input$protein_name, ",")))
      pattern <- paste0("\\b(", paste(terms, collapse = "|"), ")\\b")
      result <- result %>% filter(grepl(pattern, `Protein Name`, ignore.case = TRUE))
    }
    print(paste("Rows after Protein Name search:", nrow(result)))
    
    # Gene Name search
    if (!is.null(input$gene_name) && nzchar(input$gene_name)) {
      terms <- trimws(unlist(strsplit(input$gene_name, ",")))
      pattern <- paste0("\\b(", paste(terms, collapse = "|"), ")\\b")
      result <- result %>% filter(grepl(pattern, `Gene Name`, ignore.case = TRUE))
    }
    print(paste("Rows after Gene Name search:", nrow(result)))
    
    # PMID search
    if (!is.null(input$pmid) && nzchar(input$pmid)) {
      terms <- trimws(unlist(strsplit(input$pmid, ",")))
      pattern <- paste0("\\b(", paste(terms, collapse = "|"), ")\\b")
      result <- result %>% filter(grepl(pattern, PMID, ignore.case = TRUE))
    }
    print(paste("Rows after PMID search:", nrow(result)))
    
    # Author search
    if (!is.null(input$author) && nzchar(input$author)) {
      terms <- trimws(unlist(strsplit(input$author, ",")))
      pattern <- paste0("\\b(", paste(terms, collapse = "|"), ")\\b")
      result <- result %>% filter(grepl(pattern, Authors, ignore.case = TRUE))
    }
    print(paste("Rows after Author search:", nrow(result)))
    
    # Title / Abstract search
    if (!is.null(input$search) && nzchar(input$search)) {
      terms <- trimws(unlist(strsplit(input$search, ",")))
      pattern <- paste(terms, collapse = "|")
      result <- result %>%
        filter(grepl(pattern, Title, ignore.case = TRUE) |
                 grepl(pattern, Abstract, ignore.case = TRUE))
    }
    print(paste("Rows after Title/Abstract search:", nrow(result)))
    
    return(result)
  })

  
  # Render filtered table with abstract preview and expand button
  output$result_table <- renderDT({
    data <- filtered_data()
    
    # Cut Title and add a view button
    data$Title <- ifelse(
      nchar(data$Title) > 50,
      paste0(
        substr(data$Title, 1, 50),
        '... <button class="btn btn-link btn-sm view-btn" data-field="Title" data-text="',
        htmltools::htmlEscape(data$Title),
        '">🔍</button>'
      ),
      data$Title
    )
    
    # Cut Abstract and add a view button
    data$Abstract <- ifelse(
      nchar(data$Abstract) > 50,
      paste0(
        substr(data$Abstract, 1, 50),
        '... <button class="btn btn-link btn-sm view-btn" data-field="Abstract" data-text="',
        htmltools::htmlEscape(data$Abstract),
        '">🔍</button>'
      ),
      data$Abstract
    )
    
    datatable(
      data,
      escape = FALSE,
      options = list(
        pageLength = 10,
        lengthMenu = c(10, 25, 50),
        scrollX = TRUE,
        dom = 'tip',
        order = list(),
        columnDefs = list(
          list(targets = "_all", orderSequence = c("asc", "desc", ""))
        ),
        stateSave = FALSE
      ),
      callback = JS("
        table.on('click', '.view-btn', function() {
          var text = $(this).data('text');
          var field = $(this).data('field');
          Shiny.setInputValue('show_full_text', { field: field, text: text }, {priority: 'event'});
        });
      ")
    )
  })
  
  # Patch Notes Table Data
  patch_notes_data <- data.frame(
    Version = c("0.0.1", "0.0.2"),
    Description = c(
      paste(
        "<ul>",
        "<li>Shiny App Prototype</li>",
        "</ul>"
      ),
      paste(
        "<ul>",
        "<li>App Nickname & Logo</li>",
        "<li>Search Functionality Enhancement</li>",
        "<li>UI Cleanup</li>",
        "<li>Paper Source</li>",
        "<li>Protein Accession Handling</li>",
        "<li>Additional Tabs</li>",
        "</ul>"
      )
    ),
    Date = c("2025-05-29", "2025-06-01"),
    stringsAsFactors = FALSE
  )
  
  # Render Patch Notes Table
  output$patch_notes_table <- DT::renderDataTable({
    DT::datatable(
      patch_notes_data,
      options = list(
        pageLength = 10,
        autoWidth = TRUE
      ),
      escape = FALSE,   # <-- allow HTML rendering
      rownames = FALSE
    )
  })
  
}

# Run App
shinyApp(ui, server)
