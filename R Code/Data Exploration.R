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
vicroad_x_train = read.csv('Data/Cleaned Data/vicroad_x_train.csv', header = TRUE)
vicroad_x_val = read.csv('Data/Cleaned Data/vicroad_x_val.csv', header = TRUE)

vicroad_y_train = read.csv('Data/Cleaned Data/vicroad_y_train.csv', header = TRUE)
vicroad_y_val = read.csv('Data/Cleaned Data/vicroad_y_val.csv', header = TRUE)

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

# 2.1. Date ------------------------------------------------------------------------------------
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

accident_date_agg

# 2.2. Time ------------------------------------------------------------------------------------
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

# 2.3. Accident Type ---------------------------------------------------------------------------
vicroad_df %>%
  ggplot(aes(y = accident_type)) +
    geom_bar() +
    xlim(0, 220000) +
    geom_text(stat = "count", aes(label = after_stat(count)), hjust = -0.2, family = 'CMU Serif') +
    labs(x = 'Accident Count', y = 'Accident Type', title = 'Bar Chart of Accident Types')

# 2.4. Light Condition -------------------------------------------------------------------------
vicroad_df %>%
  ggplot(aes(y = light_condition)) +
    geom_bar() +
    geom_text(stat = "count", aes(label = after_stat(count)), hjust = -0.2, family = 'CMU Serif') +
    labs(x = 'Accident Count', y = 'Light Condition', title = 'Bar Chart of Light Condition')

# 2.5. Vehicle Year ----------------------------------------------------------------------------
vicroad_df %>%
  group_by(vehicle_year) %>%
  summarize(accident_count = n()) %>%
  ungroup() %>%
  ggplot(aes(x = vehicle_year, y = accident_count)) +
    geom_line(color = "blue", size = 0.5 ) +
    labs(
      x = "Vehicle Year",
      y = "Accident Count",
      title = 'Plot of Vehicle Year vs. Accidents'
    )




# ----------------------------------------------------------------------------------------------------------------------
# 3. Data Exploration - Bivariate Fatality Analysis
# ----------------------------------------------------------------------------------------------------------------------

# 3.1. Date ------------------------------------------------------------------------------------
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
  ggplot(aes(x = dummy_date, y = avg_percent_fatal)) +
    geom_line(colour = 'MidnightBlue', linewidth = 0.5) +
    geom_point(aes(text = paste('Date:', dummy_date, '<br>Avg Fatal:', round(avg_percent_fatal, 5))),
               alpha = 0.5) +
    scale_x_date(date_breaks = '1 month', date_labels = '%b') +
    labs(
      x = 'Month',
      y = 'Average Percentage of Fatal Accidents',
      title = 'Average Percentage of Fatal Accidents per Day (Jan–Dec)'
    )

accident_date_agg_fatal_plotly = ggplotly(accident_date_agg_fatal, tooltip = 'text')
accident_date_agg_fatal_plotly

wilcox.test(as.numeric(vicroad_df$accident_date) ~ vicroad_df$fatal)

vicroad_df %>%
  with(wilcox.test(as.numeric(as.Date(paste0('2000-', format(accident_date, '%m-%d')), format = '%Y-%m-%d')) ~ fatal))



# 3.2. Time ------------------------------------------------------------------------------------
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
    geom_line(size = 0.5, color = "red") +
    scale_x_time(
      breaks = scales::breaks_width("1 hour"),
      labels = scales::label_time("%H:%M")
    ) +
    labs(
      x      = "Time of Day",
      y      = "Percentage of Fatal Accidents (%)",
      title  = "Percentage of Fatal Accidents by Hour"
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
      values = c("Weekday" = "blue", "Weekend" = "red")
    ) +
    labs(
      x      = "Time of Day",
      y      = "Percentage of Fatal Accidents (%)",
      title  = "Percentage of Fatal Accidents by Hour",
      colour = 'Day of Week'
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

wilcox.test(as.numeric(vicroad_df$accident_time) ~ vicroad_df$fatal)

# 3.3. Accident Type ---------------------------------------------------------------------------
vicroad_df %>%
  group_by(accident_type) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(accident_type, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Accident Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Accident Type"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(accident_type != 'Other accident') %>%
  mutate(accident_type = droplevels(accident_type)) %>%
  with(chisq.test(table(accident_type, fatal)))

# 3.4. Day -------------------------------------------------------------------------------------
vicroad_df %>%
  group_by(day) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(day, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Accident Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Day"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

chisq.test(table(vicroad_df$day, vicroad_df$fatal))

# 3.5. Light Condition -------------------------------------------------------------------------
vicroad_df %>%
  group_by(light_condition) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(light_condition, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Light Condition",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Light Condition"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

chisq.test(table(vicroad_df$light_condition, vicroad_df$fatal))

# 3.6. Vehicles -------------------------------------------------------------------------
vicroad_df %>%
  mutate(vehicles_group = case_when(
    vehicles == 1 ~ "1 car",
    vehicles == 2 ~ "2 cars",
    vehicles > 2 ~ "Multi-car"
  )) %>%
  group_by(vehicles_group) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = vehicles_group, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = c("steelblue", "darkorange", "red")) +
    labs(
      x = "Number of Vehicles Involved",
      y = "Fatal Accident Percentage (%)",
      title = "Fatal Accident Percentage by Number of Vehicles"
    ) +
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

# 3.7. Persons --------------------------------------------------------------------------
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

# 3.8. Police Attend --------------------------------------------------------------------
chisq.test(table(vicroad_df$police_attend, vicroad_df$fatal))

# 3.9. Road Geometry --------------------------------------------------------------------
vicroad_df %>%
  group_by(road_geometry) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = reorder(road_geometry, -fatal_percentage), y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Road Geometry",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Road Geometry"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

chisq.test(table(vicroad_df$road_geometry, vicroad_df$fatal))


# 3.10. Speed Zone ----------------------------------------------------------------------
vicroad_df %>%
  group_by(speed_zone) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = speed_zone, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Speed Zone",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Speed Zone"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(speed_zone != 75 & speed_zone != 30) %>%
  mutate(speed_zone = droplevels(speed_zone)) %>%
  with(chisq.test(table(speed_zone, fatal)))

# 3.11. RMA -----------------------------------------------------------------------------
vicroad_df %>%
  group_by(rma) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = rma, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "RMA",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by RMA"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

vicroad_df %>%
  filter(rma != 'Non Arterial') %>%
  mutate(rma = droplevels(rma)) %>%
  with(chisq.test(table(rma, fatal)))

# 3.12. Vehicle Year --------------------------------------------------------------------
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
    geom_smooth() +
    xlim(1960, 2025) +
    ylim(0, 25)

wilcox.test(vicroad_df$vehicle_year ~ vicroad_df$fatal)

# 3.13. Initial Direction ---------------------------------------------------------------
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
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    x     = "Initial Direction",
    y     = "Fatal Percentage (%)",
    title = "Fatal Accidents by Initial Direction"
  )

chisq.test(table(vicroad_df$initial_direction, vicroad_df$fatal))

# 3.14. Road Surface --------------------------------------------------------------------
vicroad_df %>%
  group_by(road_surface) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = road_surface, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Road Surface",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Road Surface"
    )

chisq.test(table(vicroad_df$road_surface, vicroad_df$fatal))

# 3.15. Registration State --------------------------------------------------------------
vicroad_df %>%
  group_by(registration_state) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = registration_state, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Registration State",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Registration State"
    )

vicroad_df %>%
  filter(registration_state != 'NT' & registration_state != 'ACT') %>%
  mutate(registration_state = droplevels(registration_state)) %>%
  with(chisq.test(table(registration_state, fatal)))

# 3.16. Vehicle Type --------------------------------------------------------------
vicroad_df %>%
  group_by(vehicle_type) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = vehicle_type, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Vehicle Type",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Vehicle Type"
    ) +
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

# 3.17. Fuel ----------------------------------------------------------------------
vicroad_df %>%
  group_by(fuel) %>%
  summarise(fatal_percentage = mean(fatal, na.rm = TRUE) * 100) %>%
  ggplot(aes(x = fuel, y = fatal_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      x     = "Fuel",
      y     = "Fatal Percentage (%)",
      title = "Fatal Accidents by Fuel"
    )

vicroad_df %>%
  filter(fuel != 'Electric') %>%
  mutate(fuel = droplevels(fuel)) %>%
  with(chisq.test(table(fuel, fatal)))
