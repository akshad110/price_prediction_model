"""Weekly micro-insurance: rule-based risk; premium can come from ML + clamp."""


def _risk_score(rainfall: float, aqi: float, area_risk: int) -> float:
    rain_score = rainfall / 100.0
    aqi_score = aqi / 300.0
    area_score = float(area_risk)
    return (rain_score * 0.4) + (aqi_score * 0.4) + (area_score * 0.2)


def compute_risk_only(rainfall: float, aqi: float, area_risk: int) -> str:
    risk_score = _risk_score(rainfall, aqi, area_risk)
    if risk_score < 0.3:
        return "LOW"
    if risk_score < 0.6:
        return "MEDIUM"
    return "HIGH"


def compute_weekly_micro(rainfall: float, aqi: float, area_risk: int) -> tuple[str, str]:
    """Rule-based premium + risk (no Past_Disruptions)."""
    risk_score = _risk_score(rainfall, aqi, area_risk)
    risk = (
        "LOW"
        if risk_score < 0.3
        else "MEDIUM"
        if risk_score < 0.6
        else "HIGH"
    )
    base = 30.0
    premium = base + (risk_score * 40.0)
    premium_rounded = int(round(premium))
    premium_rounded = max(20, min(70, premium_rounded))
    return f"₹{premium_rounded}", risk
