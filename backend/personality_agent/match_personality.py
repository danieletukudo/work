from .database import DatabaseManager
from .loggerfile import setup_logging
import json

logger = setup_logging("database")

db = DatabaseManager()

## Example andidate test results

candidate_result = {
        "name": "Alice",
        "candidate_id": "cand_001",
        "personality": {
            "Openness": 0.8,
            "Conscientiousness": 0.9,
            "Extraversion": 0.55,
            "Agreeableness": 0.68,
            "Neuroticism": 0.25
        }}



async def calculate_match_score(candidate, job_posting_id):
    try:
        target_profile = db.get_target_personality(job_posting_id)[0]
        profile = json.loads(target_profile["target_profile"])
        print(f"target profile from db: {profile}")
        score = 0
        for trait, values in profile['target_profile'].items():
            expected = values["expected"]
            weight = values["weight"]
            candidate_score = candidate["personality"].get(trait, 0)
            diff = abs(candidate_score - expected)
            trait_score = weight * (1 - diff)  # max diff is 1
            score += trait_score
    except Exception as e:
        logger.error(f"Error calculating match score for candidate {candidate['candidate_id']} and job_posting_id {job_posting_id}: {str(e)}", exc_info=True)
        return None

    print({"name": candidate["name"], "score": round(score*100, 3)})
    score_dict = {"candidate_id": candidate["candidate_id"],
                  "name": candidate["name"],
                  "score": round(score*100, 3)}
    return score_dict