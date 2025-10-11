# Context Summary for Claude CLI Agent

## **Project Overview**
We are developing a defense against instance-based watermark forgery attacks on large language models. This builds upon prior work (Aremu et al., 2024) which introduced multi-key randomized watermarking that successfully defends against learning-based forgery attacks but acknowledged vulnerability to instance-based attacks (attacks that forge watermarks using only a single watermarked sample).

## **Core Research Question**
Can we detect instance-based watermark forgeries by exploiting the fact that single-sample attackers cannot replicate the expected distribution of watermark detection statistics (z-scores/p-values) for a specific prompt context?

## **Technical Approach**

### **Detection Algorithm:**
1. **Input:** Suspicious text of length T tokens
2. **Prefix extraction:** Extract first k tokens (k ≈ 20-50) as conditioning prompt
3. **Reference generation:** Generate n samples (n ≈ 30-50) from the provider's watermarked model using the extracted prefix
4. **Statistical measurement:**
   - Compute watermark detection z-score for suspicious text (only measuring continuation after k tokens)
   - Compute z-scores for all n regenerated samples
5. **Outlier detection:** 
   - Calculate μ = mean(legitimate_z_scores), σ = std(legitimate_z_scores)
   - If |suspicious_z_score - μ| > 3σ → classify as FORGED
   - Otherwise → classify as GENUINE

### **Key Innovation:**
Using the **first k tokens AS the prompt** (not reconstructing it) ensures perfect conditioning match between suspicious text and regenerations, eliminating reconstruction error.

## **Why This Works**
**Instance-based attackers:**
- Optimize their forgery to match ONE watermarked sample's z-score (just enough to pass threshold)
- Cannot know the prompt-specific distribution of z-scores
- Their forgery will be statistical outlier (too high or too low) compared to legitimate distribution

**Genuine texts:**
- Generated from same model/watermarking process as regenerations
- Z-score naturally falls within expected distribution

## **Experimental Objectives**

### **Primary Validation:**
1. **Measure z-score variance:** Confirm that regenerations from same prefix have tight distribution (low σ)
2. **Detect instance-based attacks:** Test on attacks by Müller et al. (2025), Jain et al. (2025) if available
3. **False positive rate:** Ensure genuine samples are not flagged (FPR < 1%)
4. **Optimal hyperparameters:** Determine best values for k (prefix length) and n (number of regenerations)

### **Secondary Analysis:**
1. **Comparison with Gloaguen et al.:** Show complementary strengths (they catch learning-based, we catch instance-based)
2. **Integration with multi-key defense:** Demonstrate combined defense catches both attack types
3. **Computational cost:** Measure wall-clock time and resource usage
4. **Robustness:** Test sensitivity to different text lengths, domains, watermarking schemes

## **Technical Stack**
- **Watermarking schemes:** KGW-SelfHash, KGW-Hard, KGW-Soft, Unigram (Kirchenbauer et al., 2023; Zhao et al., 2024)
- **Base models:** Mistral-7B (provider), Gemma-2B (surrogate attacker)
- **Detection:** Multi-key watermark detection returning z-scores for each key
- **Datasets:** C4, AdvBench, HarmfulQ, Dolly

## **Expected Outcomes**
- Z-score standard deviation < 1.0 for same-prefix regenerations
- Detection rate > 95% on instance-based forgeries
- False positive rate < 1% on genuine samples
- Clear separation in z-score distributions between genuine and forged texts

## **Implementation Tasks**
1. Implement `detect_instance_forgery(text, k, n)` function
2. Run variance analysis experiments across different k values
3. Create synthetic instance-based attack for testing
4. Generate plots comparing z-score distributions
5. Write paper comparing to Gloaguen et al. and showing complementary defense

## **Context from Prior Work**
- **Multi-key defense (Aremu et al.):** Randomize key selection during generation, detect forgery if 0 or 2+ keys detected (expects exactly 1). Reduces forgery from 100% → 2% for learning-based attacks.
- **Statistical artifact detection (Gloaguen et al.):** Detect learning-based forgeries by finding correlation artifacts. Requires 3000+ tokens, uses reprompting with original prompt.

## **Our Contribution**
A simple, effective defense against instance-based attacks that:
- Uses exact prefix (no reconstruction error)
- Works on shorter texts than Gloaguen et al.
- Complements multi-key defense (complete coverage)
- Has clear statistical foundation and interpretability

---

**In short:** We're building a statistical outlier detector that catches instance-based watermark forgeries by comparing their z-scores against the expected distribution from regenerating the same prefix. The key insight is that single-sample attackers cannot replicate prompt-specific watermark behavior.