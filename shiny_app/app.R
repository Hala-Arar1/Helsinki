library(shiny)
library(DT)
library(dplyr)
library(readr)
library(shinyjs)
library(htmltools)

# Load CSV Data
preview_df <- read.csv("data/pubmed_preview_20.csv", stringsAsFactors = FALSE)

# Placeholder
required_cols <- c(
  "Protein Name", "AC", "OS", "PMID", "Title", "Abstract", "Journal", "Authors",
  "Date Published", "Is Related to Autoregulatory", "Autoregulatory Type", "Polarity"
)

for (col in required_cols) {
  if (!(col %in% colnames(preview_df))) {
    preview_df[[col]] <- NA
  }
}

if ("PubDate" %in% colnames(preview_df)) {
  preview_df$`Date Published` <- preview_df$PubDate
}

# Mock values for UI control fields
set.seed(123)
if (all(is.na(preview_df$`Protein Name`))) {
  preview_df$`Protein Name` <- paste("Protein", seq_len(nrow(preview_df)))
}

if (all(is.na(preview_df$AC))) {
  preview_df$AC <- paste0("AC", sprintf("%04d", seq_len(nrow(preview_df))))
}

if (all(is.na(preview_df$OS))) {
  preview_df$OS <- sample(c("Homo sapiens", "Escherichia coli", "Saccharomyces cerevisiae", "Arabidopsis thaliana"), 
                          nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$Journal))) {
  preview_df$Journal <- sample(c("Nature", "Science", "Cell Reports", "J. Bacteriol"), nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$Abstract))) {
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

if (all(is.na(preview_df$`Is Related to Autoregulatory`))) {
  preview_df$`Is Related to Autoregulatory` <- sample(c("Yes", "No"), nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$`Autoregulatory Type`))) {
  preview_df$`Autoregulatory Type` <- sample(c("autoregulation", "autoinhibition", "autolysis"), nrow(preview_df), replace = TRUE)
}

if (all(is.na(preview_df$Polarity))) {
  preview_df$Polarity <- sample(c("positive", "neutral", "negative"), nrow(preview_df), replace = TRUE)
}

# Final data frame to be used
df <- preview_df[, required_cols]

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


# UI
ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$style(HTML("
      #search {
        resize: both !important;
        min-height: 80px;
        overflow: auto;
      }
      body.dark-mode {
        background-color: #121212 !important;
        color: #E0E0E0 !important;
      }
      body.dark-mode input,
      body.dark-mode select,
      body.dark-mode textarea {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border-color: #666 !important;
      }
      body.dark-mode .dataTables_wrapper {
        color: #E0E0E0 !important;
      }
      .dark-toggle {
        float: right;
        margin-top: -50px;
      }
    "))
  ),
  
  titlePanel("Biochemical Feature of Proteins"),
  actionButton("toggle_dark", "🌗 Toggle Dark Mode", class = "btn-primary dark-toggle"),
  br(), br(),
  
  fluidRow(
    column(
      width = 8,
      fluidRow(
        column(4, textInput("protein_name", "Protein Name", placeholder = "Search protein...")),
        column(4, selectInput("journal", "Journal", choices = c("All", sort(unique(na.omit(df$Journal)))), selected = "All")),
        column(4, selectInput("is_related", "Is Related to Autoregulatory", choices = c("All", sort(unique(na.omit(df$`Is Related to Autoregulatory`)))), selected = "All"))
      ),
      fluidRow(
        column(4, selectInput("type", "Autoregulatory Type", choices = c("All", sort(unique(na.omit(df$`Autoregulatory Type`)))), selected = "All")),
        column(4, selectInput("polarity", "Polarity", choices = c("All", sort(unique(na.omit(df$Polarity)))), selected = "All")),
        column(4, dateRangeInput("date_range", "Date Published",
                                 start = min(df$`Date Published`, na.rm = TRUE),
                                 end = max(df$`Date Published`, na.rm = TRUE)))
      ),
      br(),
      actionButton("reset_filters", "Reset Filters", class = "btn-warning")
    ),
    column(
      width = 4,
      textAreaInput("search", "Search Title / Abstract", placeholder = "Type or paste any text...", height = "120px")
    )
  ),
  
  DTOutput("result_table")
)

# Server
server <- function(input, output, session) {
  # Set Default Date Range
  default_date_range <- range(df$`Date Published`, na.rm = TRUE)
  
  # Toggle Dark Mode
  observeEvent(input$toggle_dark, {
    runjs("document.body.classList.toggle('dark-mode');")
  })
  
  # Reset Filters
  observeEvent(input$reset_filters, {
    updateTextInput(session, "protein_name", value = "")
    updateSelectInput(session, "journal", selected = "All")
    updateSelectInput(session, "is_related", selected = "All")
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
    
    if (input$journal != "All") {
      result <- result %>% filter(Journal == input$journal)
    }
    
    if (input$is_related != "All") {
      result <- result %>% filter(`Is Related to Autoregulatory` == input$is_related)
    }
    
    if (input$type != "All") {
      result <- result %>% filter(`Autoregulatory Type` == input$type)
    }
    
    if (input$polarity != "All") {
      result <- result %>% filter(Polarity == input$polarity)
    }
    
    if (!is.null(input$date_range)) {
      start_date <- input$date_range[1]
      end_date <- input$date_range[2]
      
      if (!is.na(start_date) && is.na(end_date)) {
        end_date <- Sys.Date()
      }
      
      if (!is.na(start_date)) {
        result <- result %>% filter(`Date Published` >= start_date)
      }
      if (!is.na(end_date)) {
        result <- result %>% filter(`Date Published` <= end_date)
      }
    }
    
    if (input$protein_name != "") {
      result <- result %>% filter(grepl(input$protein_name, `Protein Name`, ignore.case = TRUE))
    }
    
    if (input$search != "") {
      result <- result %>%
        filter(grepl(input$search, Title, ignore.case = TRUE) |
                 grepl(input$search, Abstract, ignore.case = TRUE))
    }
    
    observeEvent(input$show_full_abstract, {
      showModal(modalDialog(
        title = "Full Abstract",
        div(style = "white-space: pre-wrap; font-family: sans-serif;", input$show_full_abstract),
        easyClose = TRUE,
        footer = modalButton("Close"),
        size = "m"
      ))
    })
    
    return(result)
  })
  
  # Render Table
  output$result_table <- renderDT({
    data <- filtered_data()
    
    # Cut Abstract and add a view button
    data$Abstract <- ifelse(
      nchar(data$Abstract) > 150,
      paste0(
        substr(data$Abstract, 1, 150),
        '... <button class="btn btn-link btn-sm view-btn" data-abstract="',
        htmltools::htmlEscape(data$Abstract),
        '">🔍</button>'
      ),
      data$Abstract
    )
    
    datatable(
      data,
      escape = FALSE,
      extensions = "Buttons",
      options = list(
        pageLength = 10,
        lengthMenu = c(10, 25, 50),
        scrollX = TRUE,
        dom = 'Btip',
        buttons = list(
          list(extend = "csv", text = "Download CSV", filename = "filtered_results")
        ),
        order = list(),
        columnDefs = list(
          list(targets = "_all", orderSequence = c("asc", "desc", ""))
        ),
        stateSave = FALSE
      ),
      callback = JS("
      table.on('click', '.view-btn', function() {
        var abstractText = $(this).data('abstract');
        Shiny.setInputValue('show_full_abstract', abstractText, {priority: 'event'});
      });
    ")
    )
  })
  
}

# Run App
shinyApp(ui, server)
