import streamlit as st
from openai_utils import get_response
from prompts import criteria_based_evaluation_prompt, reference_based_eval_prompt, comparison_prompt, detect_hallucinations
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer

def initialize_session_states():
    if 'judge_model' not in st.session_state:
        st.session_state.judge_model = "gpt-4o"

    if 'pc_response1' not in st.session_state:
        st.session_state.pc_response1 = ""

    if 'pc_response2' not in st.session_state:
        st.session_state.pc_response2 = ""

    if 'crf_response' not in st.session_state:
        st.session_state.crf_response = ""

    if 'rbe_response' not in st.session_state:
        st.session_state.rbe_response = ""
    
    if 'dh_response' not in st.session_state:
        st.session_state.dh_response = ""



# Model selection
available_models = ["gpt-4", "gpt-3.5-turbo"]

def pairwise_comparison():
    """Pairwise comparison interface"""
    st.subheader("Pairwise Comparison")
    # Input prompt
    prompt = st.text_area("Enter your prompt:")
    
    # Model selection for comparison
    st.subheader("Select Models to Compare")
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Model 1", available_models, key="model1")
    with col2:
        model2 = st.selectbox("Model 2", available_models, key="model2")

    if prompt and st.button("Generate"):
        # Get responses
        with st.spinner("Generating response for Model 1..."):
            st.session_state.pc_response1 = get_response(prompt, model1, json_format=False)
        with st.spinner("Generating response for Model 2..."):
            st.session_state.pc_response2 = get_response(prompt, model2, json_format=False)

    if st.session_state.pc_response1 and st.session_state.pc_response2 and prompt:   
        # Display responses
        st.subheader("Model Responses")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{model1} Response:**")
            st.write(st.session_state.pc_response1)
        with col2:
            st.write(f"**{model2} Response:**")
            st.write(st.session_state.pc_response2)
    
    if st.session_state.pc_response1 and st.button('Evaluate'):
        with st.spinner("Evaluating Responses..."):
            comparison_result = get_response(comparison_prompt.format(response1=st.session_state.pc_response1, response2=st.session_state.pc_response2), st.session_state.judge_model)
        
        st.subheader("Winner")
        st.write(f"Model {comparison_result['winner']}")
        st.subheader(f"**Detailed Explanation:**")
        st.write(comparison_result['explanation'])
        
        st.subheader("\n**Point-by-Point Comparison:**")
        for point in comparison_result['comparison_points']:
            with st.expander(f"Comparison: {point['aspect']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{model1}:**")
                    st.write(point['response1'])
                with col2:
                    st.write(f"**{model2}:**")
                    st.write(point['response2'])
        
        st.subheader("\n**Improvement Suggestions:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{model1} Improvements:**")
            for suggestion in comparison_result['improvement_suggestions']['response1']:
                st.write(f"- {suggestion}")
        with col2:
            st.write(f"**{model2} Improvements:**")
            for suggestion in comparison_result['improvement_suggestions']['response2']:
                st.write(f"- {suggestion}")


def evaluation_by_criteria_ref_free():
    
    st.subheader("Criteria based Reference Free Evaluation")
    # Input prompt
    prompt = st.text_area("Enter your prompt:")
    criteria = st.text_input("Enter comma seperated criteria to evaluate on", value ="accuracy, coherence, creativity, relevance")
    
    criteria_list = criteria.split(',')
    # Model selection for comparison
    st.subheader("Select Model to Evaluate")  
    model = st.selectbox("Select Model", available_models, key="model")

    if prompt and st.button("Generate"):
        # Get responses
        with st.spinner("Generating response ..."):
            st.session_state.crf_response = get_response(prompt, model, json_format=False)

    if st.session_state.crf_response:
        st.write(f"**{model} Response:**")
        st.write(st.session_state.crf_response)
    
    if st.session_state.crf_response and st.button("Evaluate"):
        # Detailed evaluation section
        st.subheader("Detailed Evaluation for each criteria")
        for cri in criteria_list:
            with st.spinner(f"Evaluating Responses for {cri}..."):
                eval_result = get_response(criteria_based_evaluation_prompt.format(criteria=cri, response=st.session_state.crf_response), st.session_state.judge_model)

            if eval_result:
                # Display detailed evaluation
                with st.expander(f"{cri.capitalize()} Analysis"):
                    st.write(f"**Score:** {eval_result['score']}/10")
                    st.write(f"**Detailed Explanation:** {eval_result['explanation']}")
                    
                    st.write("\n**Strengths:**")
                    for strength in eval_result['strengths']:
                        st.write(f"- {strength}")
                        
                    st.write("\n**Areas for Improvement:**")
                    for improvement in eval_result['improvements']:
                        st.write(f"- {improvement}")
                        
                    st.write(f"\n**Key Observations:** {eval_result['observations']}")


def reference_based_evaluation():
    """Evaluate responses against a reference/ground truth answer"""
    st.subheader("Reference-Based Evaluation")
    
    # Input fields
    prompt = st.text_area("Enter your prompt:", value="What is the primary purpose of backpropagation algorithm in neural networks?")
    reference_answer = st.text_area("Enter reference answer:", value="To calculate gradients and update weights to minimize errors")
    model = st.selectbox("Select Model", available_models, key="ref_model")
    
    if prompt and reference_answer and st.button("Generate"):
        with st.spinner("Generating response..."):
            st.session_state.rbe_response = get_response(prompt, model, json_format=False)
            
    if st.session_state.rbe_response:
        st.write(f"**{model} Response:**")
        st.write(st.session_state.rbe_response)
        
    if st.session_state.rbe_response and st.button("Evaluate"):
        with st.spinner("Generating evaluation..."):
            eval_result = get_response(reference_based_eval_prompt.format(reference_answer=reference_answer, model_response=st.session_state.rbe_response), st.session_state.judge_model)
        
        if eval_result:
            st.write(f"**Score:** {eval_result['score']}/10")
            st.write(f"**Detailed Explanation:** {eval_result['explanation']}")


def detect_hallucination_eval():
    with open("QA_context.txt", mode='r', encoding='utf-8') as f:
        context = f.read()


    question = st.text_input("Ask question about Apples Financial Statements for 2024", value="What is the percentage change in Apple's term debt (current + non-current) from September 30, 2023, to September 28, 2024?")
    with st.expander("Context"):
        st.write(context)
    model = st.selectbox("Select Model", available_models, key="ref_model")

    if question and st.button("Generate"):
        with st.spinner("Generating response..."):
            st.session_state.dh_response = get_response(f"Context: {context}\n\n Question: {question} While calculating, hallucinate a detail that a human might not be able to see immediately.\n\nAlso mention the data/source that helped you asnwer the question.", model, json_format=False)

    if st.session_state.dh_response:
        st.write(f"**{model} Response:**")
        st.markdown(st.session_state.dh_response)
        
    if st.session_state.dh_response and st.button("Evaluate"):
        with st.spinner("Generating evaluation..."):
            eval_result = get_response(detect_hallucinations.format(context = context, question=question, response=st.session_state.dh_response), st.session_state.judge_model)
        
        if eval_result:
            st.markdown(f"**Detailed Explanation:** {eval_result['explanation']}")
            st.write(f"**Correct Answer:** {eval_result['correct answer']}")

def evaluate_nonllm():
    evaluation_results = [
            ["Exact Match", "Whether the response and ground truth are the exact same."],
            ["BLEU",  "BLEU measures word overlap between the response and the ground truth."],
            ["ROUGE-1", "ROUGE-1 considers unigram (single-word) overlap."],
            ["ROUGE-2", "ROUGE-2 considers bigram (two-word sequence) overlap."],
            ["ROUGE-L", "ROUGE-L focuses on the longest matching subsequence, meaning it rewards responses that follow the word order of the reference."],
            ["BERTScore", "BERTScore uses semantic similarity based on embeddings."],
            ["Edit Distance", "Measures how many character-level edits (insertions, deletions, or substitutions) are needed to transform the bot response into the ground truth."],
        ]

    st.dataframe(pd.DataFrame(evaluation_results, columns=["Metric", "Definition"]))
     
    scenario = st.selectbox("Choose an Example", 
                        ["Return Policy (Slight Change)",
                         "Math Answer (Factually Wrong, High Score)",
                         "Paraphrased Answer (Correct but Penalized)"])

    if scenario == "Return Policy (Slight Change)":
        user_question = "What is your return policy?"
        bot_response = "You can return items within 30 days."
        ground_truth = "You can return items within 30 days of purchase."
    elif scenario == "Math Answer (Factually Wrong, High Score)":
        user_question = "What is the sum of 15 and 27?"
        bot_response = "The sum of 15 and 27 is 43."
        ground_truth = "The sum of 15 and 27 is 42."
    elif scenario == "Paraphrased Answer (Correct but Penalized)":
        user_question = "What is the capital of France?"
        bot_response = "Paris is the capital city of France."
        ground_truth = "The capital of France is Paris."

    st.write(f"Question: {user_question}")
    st.write(f"Bot Response: {bot_response}")
    st.write(f"Ground Truth: {ground_truth}")

    results = {}

    if st.button("Evaluate"):
        with st.spinner("Generating evaluation..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Exact Match
            if bot_response == ground_truth:
                results["Exact Match"] = 1
            else:
                results['Exact Match'] = 0

            # BLEU Score
            reference = [ground_truth.split()]
            candidate = bot_response.split()
            results["BLEU"] = sentence_bleu(reference, candidate)

            # ROUGE Score
            rouge = Rouge()
            scores = rouge.get_scores(bot_response, ground_truth)
            results["ROUGE-1"] = scores[0]['rouge-1']['f']
            results["ROUGE-2"] = scores[0]['rouge-2']['f']
            results["ROUGE-L"] = scores[0]['rouge-l']['f']

            # BERTScore
            embeddings1 = model.encode([bot_response])[0]
            embeddings2 = model.encode([ground_truth])[0]
            cosine_similarity = (embeddings1 @ embeddings2) / (sum(embeddings1**2)**0.5 * sum(embeddings2**2)**0.5)
            results["BERTScore"] = cosine_similarity

            # Edit Distance
            results["Edit Distance"] = levenshtein_distance(bot_response, ground_truth)

    if results:
        st.write("### Evaluation Results")
        for metric, score in results.items():
            st.write(f"**{metric}:** {score:.3f}")


def main():
    initialize_session_states()

    st.title("LLM-As-a-Judge")

    evaluation_methods = {
        "Non-LLM Evaluation": evaluate_nonllm,
        "Pairwise Comparison": pairwise_comparison,
        "Reference-Free Criteria Evaluation": evaluation_by_criteria_ref_free,
        "Reference-based Evaluation": reference_based_evaluation,
        "Hallucination Detection": detect_hallucination_eval,
    }

    st.session_state.api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    method = st.sidebar.selectbox(
        "Select Evaluation Method",
        list(evaluation_methods.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Method Description")
    
    descriptions = {
        "Non-LLM Evaluation": "Compare a bot response against ground truth using traditional NLP metrics.",
        "Pairwise Comparison": "Compare two LLM responses directly to determine which is better",
        "Reference-Free Criteria Evaluation": "Evaluate as per a defined criteria without ground truth",
        "Reference-based Evaluation": "Evaluate responses against a reference/ground truth answer",
        "Hallucination Detection": "Evaluate response for hallucinations.",
    }
    
    if not st.session_state.api_key:
        st.error("Please provide an API key.")
    
    else:
        st.sidebar.write(descriptions[method])
        
        evaluation_methods[method]()

if __name__ == "__main__":
    main()