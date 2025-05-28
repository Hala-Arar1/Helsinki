# === Load Packages ===
library(shiny)
library(DT)
library(dplyr)
library(readr)

# === Load CSV Data ===
preview_df <- read.csv("../data/raw/pubmed_preview_20.csv", stringsAsFactors = FALSE)

# === Placeholder: 完整字段结构 ===
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

# === Mock values for UI control fields ===
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
if (all(is.na(preview_df$`Is Related to Autoregulatory`))) {
  preview_df$`Is Related to Autoregulatory` <- sample(c("Yes", "No"), nrow(preview_df), replace = TRUE)
}
if (all(is.na(preview_df$`Autoregulatory Type`))) {
  preview_df$`Autoregulatory Type` <- sample(c("autoregulation", "autoinhibition", "autolysis"), nrow(preview_df), replace = TRUE)
}
if (all(is.na(preview_df$Polarity))) {
  preview_df$Polarity <- sample(c("positive", "neutral", "negative"), nrow(preview_df), replace = TRUE)
}

df <- preview_df[, required_cols]

# === UI ===
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      #search {
        resize: both !important;
        min-height: 80px;
        overflow: auto;
      }
    "))
  ),
  
  titlePanel("Biochemical Feature of Proteins"),
  
  fluidRow(
    column(
      width = 8,
      fluidRow(
        column(4, selectInput("journal", "Journal", choices = c("All", sort(unique(na.omit(df$Journal)))), selected = "All")),
        column(4, selectInput("is_related", "Is Related to Autoregulatory", choices = c("All", sort(unique(na.omit(df$`Is Related to Autoregulatory`)))), selected = "All")),
        column(4, selectInput("type", "Autoregulatory Type", choices = c("All", sort(unique(na.omit(df$`Autoregulatory Type`)))), selected = "All"))
      ),
      fluidRow(
        column(4, selectInput("polarity", "Polarity", choices = c("All", sort(unique(na.omit(df$Polarity)))), selected = "All")),
        column(8, dateRangeInput("date_range", "Date Published",
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

# === Server ===
server <- function(input, output, session) {
  
  # Reset Filters 按钮逻辑
  observeEvent(input$reset_filters, {
    updateSelectInput(session, "journal", selected = "All")
    updateSelectInput(session, "is_related", selected = "All")
    updateSelectInput(session, "type", selected = "All")
    updateSelectInput(session, "polarity", selected = "All")
    updateDateRangeInput(session, "date_range",
                         start = min(df$`Date Published`, na.rm = TRUE),
                         end = max(df$`Date Published`, na.rm = TRUE))
    updateTextAreaInput(session, "search", value = "")
  })
  
  # 数据过滤逻辑
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
    
    # 应用日期范围过滤
    if (!is.null(input$date_range) &&
        !any(is.na(input$date_range)) &&
        input$date_range[1] != "" &&
        input$date_range[2] != "") {
      result <- result %>%
        filter(`Date Published` >= input$date_range[1],
               `Date Published` <= input$date_range[2])
    }
    
    # 搜索标题或摘要
    if (input$search != "") {
      result <- result %>%
        filter(grepl(input$search, Title, ignore.case = TRUE) |
                 grepl(input$search, Abstract, ignore.case = TRUE))
    }
    
    return(result)
  })
  
  # 渲染数据表格
  output$result_table <- renderDT({
    datatable(
      filtered_data(),
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
        table.on('order.dt', function () {
          var order = table.order();
          if (order.length && order[0][1] === '') {
            table.order([]).draw();
          }
        });
      ")
    )
  })
}


# === Run App ===
shinyApp(ui, server)
