#!/usr/bin/env python3
"""
Debug script to test climate clause detection logic
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

# Test the climate keyword matching logic
def test_climate_keywords():
    # Extract the climate keywords and patterns from utils.py
    climate_keywords = [
        "adaptation", "agriculture", "air pollutants", "air quality", "allergen",
        "alternative energy portfolio standard", "animal health", "asthma", "atmosphere",
        "cafe standards", "cap and trade", "cap-and-trade-program", "carbon asset risks",
        "carbon controls", "carbon dioxide", "co2", "carbon footprint", "carbon intensity",
        "carbon pollution", "carbon pollution standard", "carbon tax", "catastrophic events",
        "ch4", "changing precipitation patterns", "clean air act", "clean energy", "clean power plan",
        "climate", "climate change", "climate change regulation", "climate change risk",
        "climate disclosure", "climate issues", "climate opportunities", "climate reporting",
        "climate risk", "climate risk disclosure", "climate risks", "climate-related financial risks",
        "conference of the parties", "corporate average fuel economy", "crop failure", "droughts",
        "earthquakes", "ecosystem", "emission", "emissions", "emissions certificates",
        "emissions trading scheme", "emissions trading system", "emit", "environmental permits",
        "ets", "eu ets", "extinction", "extreme weather", "extreme weather event", "fee and remission",
        "fire season", "flooding", "fossil fuel", "fossil fuel reserves", "fossil fuels", "fuel economy",
        "ghg", "ghg emissions", "ghg regulation", "ghg trades", "global average temperature",
        "global climate", "global warming", "global warming potential", "gwp", "green",
        "green initiatives", "greenhouse effect", "greenhouse gas", "greenhouse gases",
        "gwp source", "habitat", "heat waves", "heavy precipitation", "hfcs", "high temperatures",
        "human health", "hurricanes", "hydro fluorocarbon", "infectious disease", "insured losses",
        "intended nationally determined contribution", "intergovernmental panel on climate change",
        "invasive species", "kyoto protocol", "lcfs", "low carbon fuel standard", "methane",
        "mitigation", "montreal protocol", "n2o", "natural disasters", "natural gas", "nf3",
        "nitrogen oxides", "nox", "nitrogen trifluoride", "nitrous oxide", "oil", "opportunities regulations",
        "ozone", "ozone-depleting substances", "ods", "paris agreement", "paris climate accord",
        "particulate matter", "parts per million", "per fluorocarbons", "pfcs", "persistent organic pollutants",
        "physical risks", "pollutant", "pre-industrial levels of carbon dioxide", "precipitation",
        "precipitation patterns", "rain", "rainfall", "rainwater", "regulation or disclosure of gh emissions",
        "regulatory risks", "renewable", "renewables", "renewable energy", "renewable energy goal",
        "renewable energy standard", "renewable portfolio standard", "renewable resource", "reserves",
        "risks from climate change", "risks regulations", "rps", "sea level rise", "sea-level rise",
        "sf6", "significant air emissions", "solar radiation", "sulfur oxides", "sox",
        "sulphur hexafluoride", "sustainab*", "temperatures", "ultraviolet radiation",
        "ultraviolet (uv-b) radiation", "united nations framework convention on climate change",
        "water availability", "water supply", "water vapor", "weather", "weather events",
        "weather impacts", "wildfires", "energy", "energy efficiency", "energy transition",
    ]
    
    # Test the sustainability clause
    sustainability_clause = "The Provider will use reasonable endeavours during the Term to measure and reduce greenhouse gas emissions materially associated with delivery of the Services and, on reasonable request, share a brief summary of such efforts with the Customer."
    
    print("Testing sustainability clause:")
    print(f"Text: {sustainability_clause}")
    print()
    
    # Check for exact keyword matches
    print("Checking for exact keyword matches:")
    found_keywords = []
    for keyword in climate_keywords:
        if keyword.lower() in sustainability_clause.lower():
            found_keywords.append(keyword)
    
    if found_keywords:
        print(f"Found keywords: {found_keywords}")
    else:
        print("No exact keyword matches found")
    
    print()
    
    # Check for specific phrases that should match
    print("Checking for specific climate-related phrases:")
    climate_phrases = [
        "greenhouse gas",
        "greenhouse gases", 
        "ghg",
        "emissions",
        "emission",
        "climate",
        "sustainability",
        "sustainab"
    ]
    
    for phrase in climate_phrases:
        if phrase.lower() in sustainability_clause.lower():
            print(f"✓ Found: '{phrase}'")
        else:
            print(f"✗ Missing: '{phrase}'")
    
    print()
    
    # Check if the text contains "greenhouse gas emissions" specifically
    if "greenhouse gas emissions" in sustainability_clause.lower():
        print("✓ 'greenhouse gas emissions' found in text")
    else:
        print("✗ 'greenhouse gas emissions' NOT found in text")
    
    # Check if the text contains "ghg emissions" specifically  
    if "ghg emissions" in sustainability_clause.lower():
        print("✓ 'ghg emissions' found in text")
    else:
        print("✗ 'ghg emissions' NOT found in text")

if __name__ == "__main__":
    test_climate_keywords()
