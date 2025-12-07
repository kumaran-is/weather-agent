"""Convert CSV weather data rows to narrative text for RAG.

Transforms numerical weather data into human-readable narratives that are
more suitable for semantic search and LLM retrieval.

Level 2 Implementation:
- Simple narrative templates
- City climate profiles
- Monthly/yearly summaries
- Basic aggregation

Future Enhancements (L5a):
- Advanced narrative generation with GPT
- Multi-perspective summaries
- Seasonal pattern descriptions
- Anomaly detection narratives
"""


def create_city_climate_profile(
    city: str, country: str, avg_temp: float, month: int = None, year: int = None
) -> str:
    """Create narrative description of city climate data.

    Args:
        city: City name
        country: Country name
        avg_temp: Average temperature in Fahrenheit
        month: Month number (1-12), optional
        year: Year (e.g., 2020), optional

    Returns:
        str: Narrative text describing the weather data

    Example:
        >>> text = create_city_climate_profile("Tokyo", "Japan", 68.5, 6, 2020)
        >>> print(text)
        In Tokyo, Japan, the average temperature in June 2020 was 68.5°F...
    """
    # Build time description
    if month and year:
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        time_desc = f" in {month_names[month - 1]} {year}"
    elif year:
        time_desc = f" in {year}"
    else:
        time_desc = ""

    # Temperature assessment
    if avg_temp < 32:
        temp_assessment = "freezing cold"
    elif avg_temp < 50:
        temp_assessment = "cold"
    elif avg_temp < 65:
        temp_assessment = "cool to mild"
    elif avg_temp < 75:
        temp_assessment = "mild to warm"
    elif avg_temp < 85:
        temp_assessment = "warm to hot"
    else:
        temp_assessment = "hot"

    # Build narrative
    narrative = f"In {city}, {country}, the average temperature{time_desc} was {avg_temp}°F, which is {temp_assessment}."

    # Add context for extreme temps
    if avg_temp < 10:
        narrative += " This represents severe cold conditions requiring winter weather precautions."
    elif avg_temp > 95:
        narrative += " This represents extreme heat requiring heat safety measures."

    # Add seasonal context if month provided
    if month:
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"

        narrative += f" This is typical {season} weather for the region."

    return narrative


def csv_row_to_narrative(row_dict: dict[str, any]) -> str:
    """Convert a CSV row dictionary to narrative text.

    Handles both long format (Region, Country, City, Month, Day, Year, AvgTemperature)
    and other weather data formats.

    Args:
        row_dict: Dictionary with CSV column names as keys

    Returns:
        str: Narrative description of the row

    Example:
        >>> row = {
        ...     'City': 'New York',
        ...     'Country': 'USA',
        ...     'Region': 'North America',
        ...     'Month': 7,
        ...     'Day': 15,
        ...     'Year': 2020,
        ...     'AvgTemperature': 78.5
        ... }
        >>> print(csv_row_to_narrative(row))
        In New York, USA, the average temperature in July 2020 was 78.5°F...
    """
    # Extract fields
    city = row_dict.get("City", "Unknown City")
    country = row_dict.get("Country", "Unknown Country")
    temp = row_dict.get("AvgTemperature")
    month = row_dict.get("Month")
    year = row_dict.get("Year")

    # Convert to appropriate types
    if temp is not None:
        try:
            temp = float(temp)
        except (ValueError, TypeError):
            temp = None

    if month is not None:
        try:
            month = int(month)
        except (ValueError, TypeError):
            month = None

    if year is not None:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None

    # Create narrative if we have minimum required fields
    if temp is not None:
        return create_city_climate_profile(city, country, temp, month, year)
    else:
        # Fallback: just list the fields
        parts = []
        for key, value in row_dict.items():
            if value and value != "":
                parts.append(f"{key}: {value}")
        return f"Weather record: {', '.join(parts)}"


def create_seasonal_summary(
    city: str,
    country: str,
    winter_avg: float = None,
    spring_avg: float = None,
    summer_avg: float = None,
    fall_avg: float = None,
    year: int = None,
) -> str:
    """Create narrative summary of seasonal temperature patterns.

    Args:
        city: City name
        country: Country name
        winter_avg: Average winter temperature (°F)
        spring_avg: Average spring temperature (°F)
        summer_avg: Average summer temperature (°F)
        fall_avg: Average fall temperature (°F)
        year: Year (optional)

    Returns:
        str: Narrative seasonal summary

    Example:
        >>> text = create_seasonal_summary(
        ...     "Paris", "France",
        ...     winter_avg=42.0, spring_avg=55.0,
        ...     summer_avg=72.0, fall_avg=58.0,
        ...     year=2020
        ... )
    """
    year_text = f" in {year}" if year else ""

    narrative = f"{city}, {country}{year_text} experienced seasonal temperature variations: "

    temps = []
    if winter_avg:
        temps.append(f"winter averaged {winter_avg}°F")
    if spring_avg:
        temps.append(f"spring {spring_avg}°F")
    if summer_avg:
        temps.append(f"summer {summer_avg}°F")
    if fall_avg:
        temps.append(f"fall {fall_avg}°F")

    narrative += ", ".join(temps) + "."

    # Calculate range if we have data
    all_temps = [t for t in [winter_avg, spring_avg, summer_avg, fall_avg] if t]
    if len(all_temps) >= 2:
        temp_range = max(all_temps) - min(all_temps)
        if temp_range > 50:
            narrative += f" The city shows strong seasonal variation with a {temp_range:.1f}°F range between seasons."
        elif temp_range < 20:
            narrative += f" The city has a relatively mild climate with only a {temp_range:.1f}°F range between seasons."

    return narrative


if __name__ == "__main__":
    # Test examples
    print("=" * 70)
    print("CSV to Narrative Converter - Examples")
    print("=" * 70)

    # Example 1: Single month record
    print("\nExample 1: Single Month Record")
    row1 = {
        "City": "Tokyo",
        "Country": "Japan",
        "Region": "Asia",
        "Month": 6,
        "Year": 2020,
        "AvgTemperature": 72.5,
    }
    print(csv_row_to_narrative(row1))

    # Example 2: Annual record
    print("\nExample 2: Annual Record (no month)")
    row2 = {"City": "London", "Country": "UK", "Year": 2019, "AvgTemperature": 55.0}
    print(csv_row_to_narrative(row2))

    # Example 3: Extreme cold
    print("\nExample 3: Extreme Cold")
    row3 = {
        "City": "Moscow",
        "Country": "Russia",
        "Month": 1,
        "Year": 2021,
        "AvgTemperature": 5.0,
    }
    print(csv_row_to_narrative(row3))

    # Example 4: Extreme heat
    print("\nExample 4: Extreme Heat")
    row4 = {
        "City": "Dubai",
        "Country": "UAE",
        "Month": 7,
        "Year": 2022,
        "AvgTemperature": 105.0,
    }
    print(csv_row_to_narrative(row4))

    # Example 5: Seasonal summary
    print("\nExample 5: Seasonal Summary")
    seasonal = create_seasonal_summary(
        "Paris",
        "France",
        winter_avg=42.0,
        spring_avg=55.0,
        summer_avg=75.0,
        fall_avg=58.0,
        year=2020,
    )
    print(seasonal)
