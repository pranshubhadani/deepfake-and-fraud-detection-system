import os
import sys
import link_detector
import sms_detector
import deepfake_detector

def calculate_hybrid_score(link_suspicion_pct, sms_spam_prob_pct, deepfake_artifacts_pct):
    """
    Computes the unified threat score using the project's Hybrid Verdict Formula:
    Final_Score = (Link_Suspicion * 0.4) + (SMS_Spam_Prob * 0.3) + (Deepfake_CNN_Artifacts * 0.3)
    
    All inputs should be on a 0-100 scale.
    """
    final_score = (link_suspicion_pct * 0.4) + (sms_spam_prob_pct * 0.3) + (deepfake_artifacts_pct * 0.3)
    return round(final_score, 2)

def evaluate_multi_modal_incident(text_msg=None, link_url=None, media_path=None):
    """
    Comprehensive multi-modal threat analysis across Link, SMS Text, and Deepfake Media.
    """
    results = {
        'components': {},
        'weights': {
            'link_suspicion': 0.4,
            'sms_spam': 0.3,
            'deepfake_cnn': 0.3
        }
    }
    
    # 1. SMS Message Scan
    sms_score = 0.0
    if text_msg and text_msg.strip():
        sms_model = sms_detector.get_model()
        is_scam, confidence = sms_detector.check_sms(text_msg, sms_model)
        # If it's classified safe, the spam probability is (100 - confidence)
        spam_prob = confidence if is_scam else round(100 - confidence, 2)
        sms_score = spam_prob
        results['components']['sms'] = {
            'text': text_msg,
            'is_scam': is_scam,
            'spam_probability': spam_prob,
            'status': 'Scam Detected' if is_scam else 'Normal / Safe'
        }
    else:
        results['components']['sms'] = None

    # 2. Link Detection
    link_score = 0.0
    detected_links = []
    if text_msg:
        detected_links = link_detector.extract_links_from_text(text_msg)
        
    if link_url and link_url.strip():
        if link_url not in detected_links:
            detected_links.append(link_url.strip())
            
    if detected_links:
        link_analyses = []
        max_suspicion_pct = 0.0
        for lk in detected_links:
            analysis = link_detector.analyze_link(lk)
            raw_score = analysis['score']
            # Scale raw suspicion score (0 to 5+) to 0-100%
            suspicion_pct = min(100.0, raw_score * 20.0)
            max_suspicion_pct = max(max_suspicion_pct, suspicion_pct)
            link_analyses.append({
                'url': lk,
                'status': analysis['status'],
                'raw_score': raw_score,
                'suspicion_pct': suspicion_pct,
                'reasons': analysis['reasons']
            })
        link_score = max_suspicion_pct
        results['components']['link'] = {
            'links_found': link_analyses,
            'link_suspicion_pct': link_score
        }
    else:
        results['components']['link'] = None

    # 3. Deepfake Video / Image Analysis
    deepfake_score = 0.0
    if media_path and os.path.exists(media_path):
        df_model, device = deepfake_detector.load_or_init_model()
        ext = os.path.splitext(media_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            res = deepfake_detector.analyze_video(media_path, df_model, device)
            deepfake_score = res.get('avg_fake_score', 0.0)
            results['components']['deepfake'] = res
        elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
            res = deepfake_detector.analyze_image(media_path, df_model, device)
            deepfake_score = res.get('fake_confidence', 0.0)
            results['components']['deepfake'] = res
    else:
        results['components']['deepfake'] = None

    # Calculate Hybrid Score
    final_score = calculate_hybrid_score(link_score, sms_score, deepfake_score)
    
    if final_score >= 60.0:
        overall_risk = "HIGH THREAT / FRAUD DETECTED"
        color_flag = "RED"
    elif final_score >= 35.0:
        overall_risk = "MODERATE RISK / PROCEED WITH CAUTION"
        color_flag = "YELLOW"
    else:
        overall_risk = "LOW RISK / LIKELY SAFE"
        color_flag = "GREEN"

    results['final_score'] = final_score
    results['overall_risk'] = overall_risk
    results['color_flag'] = color_flag
    return results

if __name__ == '__main__':
    print("=" * 65)
    print(" 🛡️ MULTI-MODAL SCAM & DEEPFAKE HYBRID DETECTION SYSTEM ")
    print(" Formula: Final_Score = (Link*0.4) + (SMS*0.3) + (Deepfake_CNN*0.3)")
    print("=" * 65)
    
    msg = input("\n1. Enter SMS text message (or press Enter to skip): ").strip()
    url = input("2. Enter suspect Link / URL (or press Enter to skip): ").strip()
    media = input("3. Enter path to Video / Face Image (or press Enter to skip): ").strip().strip('"').strip("'")
    
    print("\nRunning multi-modal analysis...")
    report = evaluate_multi_modal_incident(text_msg=msg, link_url=url, media_path=media)
    
    print("\n" + "=" * 50)
    print("           ANALYSIS REPORT           ")
    print("=" * 50)
    
    if report['components']['sms']:
        sms_info = report['components']['sms']
        print(f"📩 SMS Analysis:       {sms_info['status']} (Spam Prob: {sms_info['spam_probability']}%)")
        
    if report['components']['link']:
        link_info = report['components']['link']
        print(f"🔗 Link Suspicion:     {link_info['link_suspicion_pct']}%")
        for item in link_info['links_found']:
            print(f"   - {item['url']} -> {item['status']}")
            
    if report['components']['deepfake']:
        df_info = report['components']['deepfake']
        if 'avg_fake_score' in df_info:
            print(f"👁️ Deepfake Video:     {df_info['verdict']} (Score: {df_info['avg_fake_score']}%)")
        else:
            print(f"👁️ Deepfake Face:      {df_info['verdict']} (Score: {df_info['fake_confidence']}%)")
            
    print("-" * 50)
    print(f"📊 HYBRID VERDICT SCORE: {report['final_score']} / 100")
    print(f"🚨 OVERALL THREAT LEVEL: {report['overall_risk']}")
    print("=" * 50)
