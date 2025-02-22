# Globalna część aplikacji (globalny kod)
library(tidyverse)
library(ROSE)
library(caret)
library(randomForest)
library(xgboost)
library(rpart)
library(shiny)

# Wczytanie danych
df <- read.csv("creditcard.csv")
df$Class <- as.factor(df$Class)

# Oversampling
df_balanced <- ROSE(Class ~ ., data = df, seed = 123)$data

# Podział danych
set.seed(123)
trainIndex <- createDataPartition(df_balanced$Class, p = 0.7, list = FALSE)
trainData <- df_balanced[trainIndex, ]
testData  <- df_balanced[-trainIndex, ]

# Ładowanie zapisanych modeli zamiast trenowania ich na nowo
# Upewnij się, że pliki 'rf_model.rds', 'log_model.rds', 'xgb_model.rds' i 'dt_model.rds'
# znajdują się w katalogu aplikacji.
rf_model <- readRDS("rf_model.rds")
log_model <- readRDS("log_model.rds")
xgb_model <- readRDS("xgb_model.rds")
dt_model <- readRDS("dt_model.rds")

# Definicja funkcji server
server <- function(input, output) {
  
  # Sekcja Eksploracji Danych / Data Exploration Section
  observeEvent(input$refresh, {
    output$fraudPlot <- renderPlot({
      ggplot(df_balanced, aes_string(x = input$feature, fill = "Class")) +
        geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
        theme_minimal() +
        labs(title = "Rozkład zmiennej / Distribution of the Variable",
             x = input$feature,
             y = "Liczba przypadków / Count") +
        guides(fill = guide_legend(title = "Klasa / Class"))
    })
    
    output$summaryTable <- renderTable({
      summary(df_balanced)
    })
  })
  
  # Sekcja Oceny Modelu / Model Evaluation Section
  observeEvent(input$eval, {
    cm_output <- NULL
    tryCatch({
      if (input$model == "Random Forest") {
        # Domyślna metoda predict zwraca klasy
        pred_model <- predict(rf_model, testData)
        cm <- confusionMatrix(pred_model, testData$Class)
        cm_output <- capture.output(cm)
      } else if (input$model == "Logistic Regression") {
        pred_model <- predict(log_model, testData)
        cm <- confusionMatrix(pred_model, testData$Class)
        cm_output <- capture.output(cm)
      } else if (input$model == "XGBoost") {
        # Tworzymy lokalną kopię test_matrix, aby upewnić się, że DMatrix jest aktywny
        test_matrix_local <- xgb.DMatrix(data = as.matrix(testData[, !names(testData) %in% "Class"]),
                                         label = as.numeric(testData$Class) - 1)
        pred_prob <- predict(xgb_model, test_matrix_local)
        pred_model <- ifelse(pred_prob > 0.5, 1, 0)
        pred_model <- as.factor(pred_model)
        levels(pred_model) <- levels(testData$Class)
        cm <- confusionMatrix(pred_model, testData$Class)
        cm_output <- capture.output(cm)
      } else if (input$model == "Decision Tree") {
        pred_model <- predict(dt_model, testData, type = "class")
        cm <- confusionMatrix(pred_model, testData$Class)
        cm_output <- capture.output(cm)
      }
    }, error = function(e) {
      cm_output <<- paste("Błąd podczas oceny modelu:", e$message)
    })
    
    output$modelMetrics <- renderPrint({
      cat(cm_output, sep = "\n")
    })
  })
}

# Definicja interfejsu użytkownika (UI)
ui <- fluidPage(
  titlePanel("Fraud Detection in Transactions"),
  tabsetPanel(
    tabPanel("Data Exploration",
             sidebarLayout(
               sidebarPanel(
                 selectInput("feature", "Select Variable:", choices = names(df_balanced)),
                 actionButton("refresh", "Refresh")
               ),
               mainPanel(
                 plotOutput("fraudPlot"),
                 tableOutput("summaryTable")
               )
             )
    ),
    tabPanel("Model Evaluation",
             sidebarLayout(
               sidebarPanel(
                 selectInput("model", "Select Model:", 
                             choices = c("Random Forest", "Logistic Regression", "XGBoost", "Decision Tree")),
                 actionButton("eval", "Evaluate")
               ),
               mainPanel(
                 verbatimTextOutput("modelMetrics")
               )
             )
    )
  )
)


# Uruchomienie aplikacji / Run the Shiny App
shinyApp(ui, server)
