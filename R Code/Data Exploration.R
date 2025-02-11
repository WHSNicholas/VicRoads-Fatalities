# -------------------------------------------------------------------------------------------------------------------- #
#                                           VicRoads Motor Fatalities Model                                            #
#                                              Exploratory Data Analysis                                               #
# -------------------------------------------------------------------------------------------------------------------- #

# Title: VicRoads Motor Fatalities Model
# Script: Exploratory Data Analysis
#
# Authors: Nicholas Wong
# Creation Date: 11th December 2024
# Modification Date: 11th December 2024
#
# Purpose: This script explores the relationship between road fatalities and driver/accident profiles. We use data
#          publicly available from VicRoad to perform data analysis as well as preparing the data for the development of
#          a predictive model using machine learning techniques.
#
# Dependencies: pandas
#
# Instructions: Ensure that the working directory is set to VicRoads-Fatalities
#
# Data Sources: VicRoad Data obtained from https://discover.data.vic.gov.au/dataset/victoria-road-crash-data
# - Accident Data
# - Vehicle Data
# - Accident Event Data
# - Atmospheric Condition Data
# - Sub DCA Data
# - Person Data
# - Node Data
# - Road Surface Condition Data
# - Accident Location Data
#
# Fonts: 'CMU Serif.ttf'
#
# Table of Contents:
# 1. Data Integration
#   1.1. Preamble
#   1.2. Importing Data
# 2. Data Cleaning
#   2.1. Structuring Data
#   2.2. NA Cleaning
# 3. Data Exploration
# 4. Data Transformation
#   4.1. Transformation
#   4.2. Dimension Reduction


# ----------------------------------------------------------------------------------------------------------------------
# 1. Data Cleaning
# ----------------------------------------------------------------------------------------------------------------------

# 1.1. Preamble ----------------------------------------------------------------------------------------------
# Required Packages
library(tidyverse)
library(lubridate)
library(plotly)
library(hms)
library(RColorBrewer)
library(sf)

# Settings
options(max.print = 10000)
options(scipen = 999)

# Theme
theme_set(
  theme_grey() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16),
      axis.line = element_line(linewidth = 1, colour = 'grey80'),
      text = element_text(family = 'CMU Serif', size = 12)
    )
)

# 1.2. Importing Cleaned Data --------------------------------------------------------------------------------
# VicRoad Data
vicroad_x_train = read.csv('Data/Cleaned Data/vicroad_x_train.csv', header = TRUE)
vicroad_x_val = read.csv('Data/Cleaned Data/vicroad_x_val.csv', header = TRUE)

vicroad_y_train = read.csv('Data/Cleaned Data/vicroad_y_train.csv', header = TRUE)
vicroad_y_val = read.csv('Data/Cleaned Data/vicroad_y_val.csv', header = TRUE)

# Shapefile Data
victoria_sf = st_read("Data/Shapefiles/MB_2016_VIC.shp")

# Forrming full dataset
vicroad_x = rbind(vicroad_x_train, vicroad_x_val)
vicroad_y = rbind(vicroad_y_train, vicroad_y_val)

vicroad_y$fatal = ifelse(vicroad_y$persons_killed > 0, TRUE, FALSE)

vicroad_df = cbind(vicroad_x, vicroad_y)

rm(vicroad_x_train, vicroad_x_val, vicroad_y_train, vicroad_y_val, vicroad_x, vicroad_y)

# 1.3. Summary Statistics ------------------------------------------------------------------------------------
# Factors
factors = c(
    'accident_type', 'day', 'dca_code', 'light_condition', 'road_geometry', 'severity', 'rma', 'vehicle_dca_code', 'speed_zone',
    'initial_direction', 'road_surface', 'registration_state', 'vehicle_body_style', 'vehicle_make', 'vehicle_model',
    'vehicle_type', 'fuel', 'final_direction', 'driver_intent', 'vehicle_movement', 'trailer_type', 'caught_fire',
    'initial_impact', 'lamps', 'traffic_control', 'sex', 'age_group', 'inj_level', 'helmet_belt_worn', 'road_user',
    'license_state', 'node_type', 'lga_name', 'lga_name_all', 'deg_urban_name', 'postcode_crash', 'road_route',
    'road_name', 'road_type', 'road_name_intersection', 'road_type_intersection', 'direction_location'
)

vicroad_df = vicroad_df %>%
  mutate(
    across(all_of(factors), as.factor
    )
  )

# Datetime
vicroad_df = vicroad_df %>% mutate(across(c(accident_date), ymd))
vicroad_df = vicroad_df %>% mutate(across(c(accident_time), as_hms))

# Boolean
boolean = names(vicroad_df)[grepl(
  '^(event_type|veh_1_coll|veh_2_coll|atmosph_cond|sub_dca_code|surface_cond)',
  names(vicroad_df)
)]

boolean = c(boolean, 'police_attend', 'towed_away', 'ejected', 'taken_hospital')

vicroad_df <- vicroad_df %>%
  mutate(across(all_of(boolean), ~ . == 'True'))

# Summary Statistics
summary(vicroad_df)
glimpse(vicroad_df)
colSums(is.na(vicroad_df))


# ----------------------------------------------------------------------------------------------------------------------
# 2. Data Exploration - Univariate Analysis
# ----------------------------------------------------------------------------------------------------------------------
# 2.1. Date --------------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(accident_date) %>%
  summarise(accident_count = n()) %>%
  ggplot(aes(x = accident_date, y = accident_count)) +
    geom_point(alpha = 0.2, size = 1) +
    geom_smooth(colour = 'MidnightBlue', fill = 'lightblue', linewidth = 0.5) +
    scale_x_date(date_breaks = '1 year', date_labels = '%Y') +
    labs(x = 'Accident Date', y = 'Accident Count', title = 'Number of Accidents Per Day')

vicroad_df %>%
  filter(accident_date >= as.Date('2020-01-01') & accident_date <= as.Date('2023-12-31')) %>%  # Filter for COVID years
  group_by(accident_date) %>%
  summarise(accident_count = n()) %>%
  ggplot(aes(x = accident_date, y = accident_count)) +
    geom_point(alpha = 0.2, size = 1) +
    geom_smooth(colour = 'MidnightBlue', fill = 'lightblue', linewidth = 0.5) +
    scale_x_date(date_breaks = '1 year', date_labels = '%Y') +
    labs(x = 'Accident Date', y = 'Accident Count', title = 'Number of Accidents Per Day (COVID Years)')

accident_date_agg = vicroad_df %>%
  # Count of Accidents per day
  group_by(accident_date) %>%
  summarise(accident_count = n()) %>%
  ungroup() %>%

  # Creating a dummy date
  mutate(
    month_day = format(accident_date, '%m-%d'),
    dummy_date = as.Date(paste0('2000-', month_day), format = '%Y-%m-%d')
  ) %>%

  # Group by the dummy date
  group_by(dummy_date) %>%
  summarise(avg_accident = mean(accident_count)) %>%
  ungroup() %>%

  ggplot(aes(x = dummy_date, y = avg_accident)) +
    geom_line(colour = 'MidnightBlue', linewidth = 0.5) +
    geom_point(aes(text = paste('Date:', dummy_date, '<br>Avg Accidents:', avg_accident)), alpha = 0.5) +
  scale_x_date(date_breaks = '1 month', date_labels = '%b') +
  labs(
    x = 'Month',
    y = 'Average Accident Count',
    title = 'Average Number of Accidents per Day (Jan–Dec)'
  )

accident_date_agg_plotly = ggplotly(accident_date_agg, tooltip = 'text')
accident_date_agg_plotly

htmlwidgets::saveWidget(accident_date_agg_plotly, "accident_date_agg_plotly.html", selfcontained = FALSE)

accident_date_agg

# 2.2. Time --------------------------------------------------------------------------------------------------
vicroad_df %>%
  mutate(
    accident_time_sec    = as.numeric(accident_time),
    accident_time_15_sec = floor(accident_time_sec / 900) * 900
  ) %>%
  # Group by day_category and the 15-minute interval
  group_by(accident_time_15_sec) %>%
  summarise(
    accident_count = n() / n_distinct(accident_date),
    .groups = "drop"
  ) %>%
  mutate(accident_time_15 = as_hms(accident_time_15_sec)) %>%
  ggplot(aes(x = accident_time_15, y = accident_count)) +
    geom_line(size = 0.5) +
    scale_x_time(
      breaks = scales::breaks_width("1 hour"),
      labels = scales::label_time("%H:%M")
    ) +
    labs(
      x      = "Time of Day",
      y      = "Accident Count",
      title  = "Accidents by Hour"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  mutate(
    # Categorize days as 'Weekday' or 'Weekend'
    day_category = ifelse(day %in% c("Saturday", "Sunday"), "Weekend", "Weekday"),
    accident_time_sec    = as.numeric(accident_time),
    accident_time_15_sec = floor(accident_time_sec / 900) * 900
  ) %>%
  # Group by day_category and the 15-minute interval
  group_by(day_category, accident_time_15_sec) %>%
  summarise(
    accident_count = n() / n_distinct(accident_date),
    .groups = "drop"
  ) %>%
  mutate(accident_time_15 = as_hms(accident_time_15_sec)) %>%
  ggplot(aes(x = accident_time_15, y = accident_count, color = day_category, group = day_category)) +
    geom_line(size = 0.5) +
    scale_x_time(
      breaks = scales::breaks_width("1 hour"),
      labels = scales::label_time("%H:%M")
    ) +
    scale_color_manual(
      values = c("Weekday" = "MidnightBlue", "Weekend" = "SteelBlue")
    ) +
    labs(
      x      = "Time of Day",
      y      = "Accident Count",
      title  = "Accidents by Hour",
      colour = 'Day of Week'
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2.3. Accident Type -----------------------------------------------------------------------------------------
vicroad_df %>%
  ggplot(aes(y = accident_type)) +
    geom_bar() +
    xlim(0, 220000) +
    geom_text(stat = "count", aes(label = after_stat(count)), hjust = -0.2, family = 'CMU Serif') +
    labs(x = 'Accident Count', y = 'Accident Type', title = 'Bar Chart of Accident Types')

# 2.4. Light Condition ---------------------------------------------------------------------------------------
vicroad_df %>%
  ggplot(aes(y = light_condition)) +
    geom_bar() +
    geom_text(stat = "count", aes(label = after_stat(count)), hjust = -0.2, family = 'CMU Serif') +
    labs(x = 'Accident Count', y = 'Light Condition', title = 'Bar Chart of Light Condition')

# 2.5. Vehicle Year ------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(vehicle_year) %>%
  summarize(accident_count = n()) %>%
  ungroup() %>%
  ggplot(aes(x = vehicle_year, y = accident_count)) +
    geom_line(color = "Midnight Blue", size = 0.5 ) +
    labs(
      x = "Vehicle Year",
      y = "Accident Count",
      title = 'Plot of Vehicle Year vs. Accidents'
    )




# ----------------------------------------------------------------------------------------------------------------------
# 3. Data Exploration - Bivariate Fatality Analysis
# ----------------------------------------------------------------------------------------------------------------------
# 3.1. Date --------------------------------------------------------------------------------------------------
accident_date_agg_fatal = vicroad_df %>%
  # Group by accident_date and calculate total accidents and fatal accidents
  group_by(accident_date) %>%
  summarise(
    total_accidents = n(),
    fatal_accidents = sum(fatal)
  ) %>%
  ungroup() %>%

  # Calculate percentage of fatal accidents for each day
  mutate(percent_fatal = fatal_accidents / total_accidents) %>%

  # Create dummy date for averaging percentages across years
  mutate(
    month_day = format(accident_date, '%m-%d'),
    dummy_date = as.Date(paste0('2000-', month_day), format = '%Y-%m-%d')
  ) %>%

  # Group by the dummy date and calculate average percentage of fatal accidents
  group_by(dummy_date) %>%
  summarise(avg_percent_fatal = mean(percent_fatal, na.rm = TRUE)) %>%
  ungroup() %>%

  # Plot the data
  ggplot(aes(x = dummy_date, y = avg_percent_fatal * 100)) +
    geom_line(colour = 'MidnightBlue', linewidth = 0.5) +
    geom_point(aes(text = paste('Date:', dummy_date, '<br>Avg Fatal:', round(avg_percent_fatal, 5))),
               alpha = 0.5) +
    scale_x_date(date_breaks = '1 month', date_labels = '%b') +
    labs(
      x = 'Month',
      y = 'Average Percentage of Fatal Accidents',
      title = 'Average Percentage of Fatal Accidents per Day (Jan–Dec)'
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

accident_date_agg_fatal_plotly = ggplotly(accident_date_agg_fatal, tooltip = 'text')
accident_date_agg_fatal_plotly

wilcox.test(as.numeric(vicroad_df$accident_date) ~ vicroad_df$fatal)

vicroad_df %>%
  with(wilcox.test(as.numeric(as.Date(paste0('2000-', format(accident_date, '%m-%d')), format = '%Y-%m-%d')) ~ fatal))

# 3.2. Time --------------------------------------------------------------------------------------------------
vicroad_df %>%
  mutate(
    accident_time_sec    = as.numeric(accident_time),
    accident_time_15_sec = floor(accident_time_sec / 900) * 900
  ) %>%
  # Group by the 15-minute interval
  group_by(accident_time_15_sec) %>%
  summarise(
    fatal_percentage = mean(fatal, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  mutate(accident_time_15 = as_hms(accident_time_15_sec)) %>%
  ggplot(aes(x = accident_time_15, y = fatal_percentage)) +
    geom_line(size = 0.5, color = "MidnightBlue") +
    scale_x_time(
      breaks = scales::breaks_width("1 hour"),
      labels = scales::label_time("%H:%M")
    ) +
    labs(
      x      = "Time of Day",
      y      = "Percentage of Fatal Accidents (%)",
      title  = "Percentage of Fatal Accidents by Hour"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  mutate(
    # Categorize days as 'Weekday' or 'Weekend'
    day_category = ifelse(day %in% c("Saturday", "Sunday"), "Weekend", "Weekday"),
    accident_time_sec    = as.numeric(accident_time),
    accident_time_15_sec = floor(accident_time_sec / 900) * 900
  ) %>%
  # Group by day_category and the 15-minute interval
  group_by(day_category, accident_time_15_sec) %>%
  summarise(
    fatal_percentage = mean(fatal, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  mutate(accident_time_15 = as_hms(accident_time_15_sec)) %>%
  ggplot(aes(x = accident_time_15, y = fatal_percentage, color = day_category, group = day_category)) +
    geom_line(size = 0.5) +
    scale_x_time(
      breaks = scales::breaks_width("1 hour"),
      labels = scales::label_time("%H:%M")
    ) +
    scale_color_manual(
      values = c("Weekday" = "MidnightBlue", "Weekend" = "SteelBlue")
    ) +
    labs(
      x      = "Time of Day",
      y      = "Percentage of Fatal Accidents (%)",
      title  = "Percentage of Fatal Accidents by Hour",
      colour = 'Day of Week'
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

wilcox.test(as.numeric(vicroad_df$accident_time) ~ vicroad_df$fatal)

# 3.3. Accident Type -----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(accident_type) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(accident_type, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Accident Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Accident Type"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(accident_type != 'Other accident') %>%
  mutate(accident_type = droplevels(accident_type)) %>%
  with(chisq.test(table(accident_type, fatal)))

# 3.4. Day ---------------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(day) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(day, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Accident Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Day"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

chisq.test(table(vicroad_df$day, vicroad_df$fatal))

# 3.5. Light Condition ---------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(light_condition) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(light_condition, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Light Condition",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Light Condition"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

chisq.test(table(vicroad_df$light_condition, vicroad_df$fatal))

# 3.6. Vehicles ----------------------------------------------------------------------------------------------
vicroad_df %>%
  mutate(vehicles_group = case_when(
    vehicles == 1 ~ "1 car",
    vehicles == 2 ~ "2 cars",
    vehicles > 2 ~ "Multi-car"
  )) %>%
  group_by(vehicles_group) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = vehicles_group, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x = "Number of Vehicles Involved",
      y = "Fatal Accident Percentage (%)",
      title = "Fatal Accident Percentage by Number of Vehicles"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

vicroad_df %>%
  mutate(vehicles_group = case_when(
    vehicles == 1 ~ "1 car",
    vehicles == 2 ~ "2 cars",
    vehicles == 3 ~ "3 cars",
    vehicles == 4 ~ "4 cars",
    vehicles == 5 ~ "5 cars",
    vehicles == 6 ~ "6 cars",
    vehicles > 6  ~ "7+ cars"
  )) %>%
  with(chisq.test(table(vehicles_group, fatal)))

# 3.7. Persons -----------------------------------------------------------------------------------------------
vicroad_df %>%
  mutate(persons_group = case_when(
    persons == 1 ~ "1 person",
    persons == 2 ~ "2 persons",
    persons == 3 ~ "3 persons",
    persons == 4 ~ "4 persons",
    persons == 5 ~ "5 persons",
    persons == 6 ~ "6 persons",
    persons > 6 ~ "6+ persons"
  )) %>%
  group_by(persons_group) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = persons_group, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x = "Number of Vehicles Involved",
      y = "Fatal Accident Percentage (%)",
      title = "Fatal Accident Percentage by Number of Vehicles"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

vicroad_df %>%
  mutate(persons_group = case_when(
    persons == 1 ~ "1 person",
    persons == 2 ~ "2 persons",
    persons == 3 ~ "3 persons",
    persons == 4 ~ "4 persons",
    persons == 5 ~ "5 persons",
    persons == 6 ~ "6 persons",
    persons > 6  ~ "7+ persons"
  )) %>%
  with(chisq.test(table(persons_group, fatal)))

# 3.8. Police Attend -----------------------------------------------------------------------------------------
chisq.test(table(vicroad_df$police_attend, vicroad_df$fatal))

# 3.9. Road Geometry -----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(road_geometry) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(road_geometry, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Road Geometry",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Road Geometry"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

chisq.test(table(vicroad_df$road_geometry, vicroad_df$fatal))

# 3.10. Speed Zone -------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(speed_zone) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = speed_zone, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Speed Zone",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Speed Zone"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(speed_zone != 75 & speed_zone != 30) %>%
  mutate(speed_zone = droplevels(speed_zone)) %>%
  with(chisq.test(table(speed_zone, fatal)))

# 3.11. RMA --------------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(rma) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = rma, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "RMA",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by RMA"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(rma != 'Non Arterial') %>%
  mutate(rma = droplevels(rma)) %>%
  with(chisq.test(table(rma, fatal)))

# 3.12. Vehicle Year -----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(vehicle_year) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = vehicle_year, y = fatal_percentage)) +
    geom_line(linewidth = 0.5) +
    labs(
      x     = "Vehicle Year",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Vehicle Year"
    ) +
    geom_smooth(fill = 'MidnightBlue') +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    xlim(1960, 2025) +
    ylim(0, 25)

wilcox.test(vicroad_df$vehicle_year ~ vicroad_df$fatal)

# 3.13. Initial Direction ------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(initial_direction) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  mutate(
    initial_direction = factor(
      initial_direction,
      levels = c("N", "NE", "E", "SE", "S", 'SW', 'W', 'NW', 'NK')
    )
  ) %>%
  ggplot(aes(x = initial_direction, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Initial Direction",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Initial Direction"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

chisq.test(table(vicroad_df$initial_direction, vicroad_df$fatal))

# 3.14. Road Surface -----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(road_surface) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = road_surface, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Road Surface",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Road Surface"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

chisq.test(table(vicroad_df$road_surface, vicroad_df$fatal))

# 3.15. Registration State -----------------------------------------------------------------------------------
vicroad_df %>%
  group_by(registration_state) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = registration_state, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Registration State",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Registration State"
    )

vicroad_df %>%
  filter(registration_state != 'NT' & registration_state != 'ACT') %>%
  mutate(registration_state = droplevels(registration_state)) %>%
  with(chisq.test(table(registration_state, fatal)))

# 3.16. Vehicle Type -----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(vehicle_type) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = vehicle_type, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Vehicle Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Vehicle Type"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(
  vehicle_type != 'Electric Device' &
  vehicle_type != 'Moped' &
  vehicle_type != 'Prime Mover (No of Trailers Unknown)' &
  vehicle_type != 'Rigid Truck(Weight Unknown)'
  ) %>%
  mutate(vehicle_type = droplevels(vehicle_type)) %>%
  with(chisq.test(table(vehicle_type, fatal)))

# 3.17. Fuel -------------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(fuel) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = fuel, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Fuel",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Fuel"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  filter(fuel != 'Electric') %>%
  mutate(fuel = droplevels(fuel)) %>%
  with(chisq.test(table(fuel, fatal)))

# 3.18. Wheels -----------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(wheels) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = wheels, y = fatal_percentage)) +
    geom_point() +
    labs(
      x     = "Wheels",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Wheels"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

wilcox.test(vicroad_df$wheels ~ vicroad_df$fatal)

# 3.19. Cylinders --------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(cylinders) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = cylinders, y = fatal_percentage)) +
    geom_point() +
    labs(
      x     = "Cylinders",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Cylinders"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

wilcox.test(vicroad_df$cylinders ~ vicroad_df$fatal)

# 3.20. Seating Capacity -------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(seating_capacity) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = seating_capacity, y = fatal_percentage)) +
    geom_point() +
    labs(
      x     = "Seating Capacity",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Seating Capacity"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

wilcox.test(vicroad_df$seating_capacity ~ vicroad_df$fatal)

# 3.22. Tare Weight ------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(tare_weight) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = tare_weight, y = fatal_percentage)) +
    geom_point() +
    labs(
      x     = "Tare Weight",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Tare Weight"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

wilcox.test(vicroad_df$tare_weight ~ vicroad_df$fatal)

# 3.23. Final Direction --------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(final_direction) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  mutate(
    initial_direction = factor(
      final_direction,
      levels = c("N", "NE", "E", "SE", "S", 'SW', 'W', 'NW', 'NK')
    )
  ) %>%
  ggplot(aes(x = final_direction, y = fatal_percentage)) +
  geom_bar(stat = "identity") +
  labs(
    x     = "Final Direction",
    y     = "Fatal Percentage (%)",
    title = "Fatal Accidents by Final Direction"
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1))

chisq.test(table(vicroad_df$final_direction, vicroad_df$fatal))

# 3.24. Driver Intent ----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(driver_intent) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(driver_intent, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Driver Intent",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Driver Intent"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(driver_intent != 'Parking illegally' & driver_intent != 'Parking or unparking') %>%
  mutate(driver_intent = droplevels(driver_intent)) %>%
  with(chisq.test(table(driver_intent, fatal)))

# 3.24. Vehicle Movement -------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(vehicle_movement) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(vehicle_movement, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Vehicle Movement",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Vehicle Movement"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(vehicle_movement != 'Parking illegally' & vehicle_movement != 'Parking or unparking') %>%
  mutate(vehicle_movement = droplevels(vehicle_movement)) %>%
  with(chisq.test(table(vehicle_movement, fatal)))

# 3.25. Trailer Type -----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(trailer_type) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(trailer_type, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Trailer Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Trailer Type"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(trailer_type != 'Machinery') %>%
  mutate(trailer_type = droplevels(trailer_type)) %>%
  with(chisq.test(table(trailer_type, fatal)))

# 3.26. Caught Fire ------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(caught_fire) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(caught_fire, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Caught Fire",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Caught Fire"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  with(chisq.test(table(caught_fire, fatal)))

# 3.26. Initial Impact ---------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(initial_impact) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(initial_impact, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Initial Impact",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Initial Impact"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(initial_impact != 'Sidecar') %>%
  mutate(initial_impact = droplevels(initial_impact)) %>%
  with(chisq.test(table(initial_impact, fatal)))

# 3.27. Lamps ------------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(lamps) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(lamps, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Lamps",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Lamps"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  with(chisq.test(table(lamps, fatal)))


# 3.28. Traffic Control --------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(traffic_control) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(traffic_control, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Traffic Control",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Traffic Control"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(
    traffic_control != 'Flashing lights' &
    traffic_control != 'Out of order' &
    traffic_control != 'Police' &
    traffic_control != 'RX Bells/Lights' &
    traffic_control != 'RX No control' &
    traffic_control != 'School Flags' &
    traffic_control != 'School No flags'
  ) %>%
  mutate(traffic_control = droplevels(traffic_control)) %>%
  with(chisq.test(table(traffic_control, fatal)))

# 3.29. Event Type -------------------------------------------------------------------------------------------
vicroad_df %>%
  pivot_longer(
    cols = starts_with("event_type"),
    names_to = "event_type",
    values_to = "event_occurred"
  ) %>%
  filter(event_occurred == 1) %>%
  group_by(event_type) %>%
  summarise(
    fatal_count = sum(fatal),
    total_count = n(),
    fatal_percentage = 100 * fatal_count / total_count,
    .groups = "drop"
  ) %>%
ggplot(aes(x = event_type, y = fatal_percentage)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(
    title = "Percentage of Fatal Accidents by Event Type",
    x = "Event Type",
    y = "Fatal Percentage (%)"
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 3.30. Vehicle Collision ------------------------------------------------------------------------------------
vicroad_df %>%
  pivot_longer(
    cols = starts_with("veh_"),
    names_to = "collision_type",
    values_to = "collision_occurred"
  ) %>%
  filter(collision_occurred == 1) %>%
  mutate(collision_type = sub("veh_\\d+_coll_", "", collision_type)) %>%
  group_by(collision_type) %>%
  summarise(
    fatal_count = sum(fatal),
    total_count = n(),
    fatal_percentage = 100 * fatal_count / total_count,
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = reorder(collision_type, -fatal_percentage),
    y = fatal_percentage
  )) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    labs(
      title = "Average Fatal Percentage by Collision Type",
      x = "Collision Type",
      y = "Fatal Percentage (%)"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 3.31. Sub DCA Code -----------------------------------------------------------------------------------------
vicroad_df %>%
  pivot_longer(
    cols = starts_with("sub_dca_code"),
    names_to = "sub_dca_type",
    values_to = "occured"
  ) %>%
  filter(occured == 1) %>%
  mutate(sub_dca_type = sub("sub_dca_code_", "", sub_dca_type)) %>%
  group_by(sub_dca_type) %>%
  summarise(
    fatal_count = sum(fatal),
    total_count = n(),
    fatal_percentage = 100 * fatal_count / total_count,
    .groups = "drop"
  ) %>%
  ggplot(aes(
    y = reorder(sub_dca_type, -fatal_percentage),
    x = fatal_percentage,
    fill = reorder(sub_dca_type, -fatal_percentage)
  )) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    labs(
      title = "Average Fatal Percentage by Collision Type",
      y = "Collision Type",
      x = "Fatal Percentage (%)"
    ) +
    scale_x_continuous(labels = scales::percent_format(scale = 1))

# 3.32. Sex --------------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(sex) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(sex, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Sex",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Sex"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  with(chisq.test(table(sex, fatal)))

# 3.33. Age Group --------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(age_group) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(age_group, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Age Group",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Age Group"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  with(chisq.test(table(age_group, fatal)))

# 3.33. Helmet Belt Worn -------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(helmet_belt_worn) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(helmet_belt_worn, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Helmet Belt Worn",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Helmet Belt Worn"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  with(chisq.test(table(helmet_belt_worn, fatal)))

# 3.34. Road User --------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(road_user) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(road_user, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Road User",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Road User"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
    filter(
    road_user != 'E-scooter Rider' &
    road_user != 'Not Known' &
    road_user != 'Passengers'
  ) %>%
  mutate(road_user = droplevels(road_user)) %>%
  with(chisq.test(table(road_user, fatal)))

# 3.35. License State ----------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(license_state) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = license_state, y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "License State",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by License State"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  filter(license_state != 'NT' & license_state != 'ACT') %>%
  mutate(license_state = droplevels(license_state)) %>%
  with(chisq.test(table(license_state, fatal)))

# 3.36. Taken Hospital ---------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(taken_hospital) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(taken_hospital, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Taken Hospital",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Taken Hospital"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  with(chisq.test(table(taken_hospital, fatal)))

# 3.37. Ejected ----------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(ejected) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(ejected, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Ejected",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Ejected"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  with(chisq.test(table(ejected, fatal)))

# 3.37. Node Type --------------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(node_type) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(node_type, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity") +
    labs(
      x     = "Node Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Node Type"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1))

vicroad_df %>%
  with(chisq.test(table(node_type, fatal)))

# 3.38. AMG --------------------------------------------------------------------------------------------------
vicroad_sf = st_as_sf(
  vicroad_df,
  coords = c("amg_x", "amg_y"),
  crs = 3111
)

vicroad_wgs84 = st_transform(vicroad_sf, crs = 4326)
victoria_sf = st_transform(victoria_sf, crs = st_crs(victoria_sf))

ggplot() +
  geom_sf(data = victoria_sf, fill = "white", color = "black") +
  geom_sf(data = vicroad_sf, aes(color = fatal), size = 1, alpha = 0.5, shape = 2) +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "black")) +
  labs(
    title = "Fatal Accidents in Victoria (VicGrid94)",
    x = 'Latitude',
    y = 'Longitude',
    colour = 'Fatal'
  )


ggplot() +
  geom_sf(data = victoria_sf, fill = "white", color = "black") +
  geom_sf(
    data = vicroad_sf,
    aes(shape = fatal, color = fatal, alpha = fatal),
    size = 1
  ) +
  scale_shape_manual(values = c("TRUE" = 17, "FALSE" = 1)) +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "black")) +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.1)) +
  labs(
    title = "Fatal Accidents in Victoria (VicGrid94)",
    x = 'Latitude',
    y = 'Longitude',
    colour = 'Fatal',
    shape = 'Fatal'
  )

# 3.39. Surface Condition ------------------------------------------------------------------------------------
vicroad_df %>%
  pivot_longer(
    cols = starts_with("surface_cond"),
    names_to = "surface_cond",
    values_to = "dummy"
  ) %>%
  filter(dummy == 1) %>%
  group_by(surface_cond) %>%
  summarise(
    fatal_count = sum(fatal),
    total_count = n(),
    fatal_percentage = 100 * fatal_count / total_count,
    .groups = "drop"
  ) %>%
ggplot(aes(x = surface_cond, y = fatal_percentage) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(
    title = "Percentage of Fatal Accidents by Surface Condition",
    x = "Surface Condition",
    y = "Fatal Percentage (%)"
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
