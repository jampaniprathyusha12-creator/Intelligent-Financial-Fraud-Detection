

def calculate_risk(probability):

    """
    Classify transaction risk using fraud probability.
    """

    if probability < 0.30:
        return {
            "risk_level": "Low Risk",
            "recommendation": "Transaction appears normal."
        }

    elif probability < 0.70:
        return {
            "risk_level": "Medium Risk",
            "recommendation": "Transaction requires additional verification."
        }

    else:
        return {
            "risk_level": "High Risk",
            "recommendation": "Potential fraud detected. Generate an alert."
        }


def generate_alert(probability, risk_level):

    if risk_level == "High Risk":
        return "HIGH PRIORITY: Suspicious transaction detected!"

    elif risk_level == "Medium Risk":
        return "WARNING: Transaction needs further verification."

    return "No immediate fraud alert."
