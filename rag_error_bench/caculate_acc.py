import json
from rouge import Rouge

# 在文件开始处初始化所有计数器
num_samples = 0
parse_failed_count = 0  # 确保这个变量在最开始就被初始化

# 定义tag1和tag2的有效类别
VALID_TAG1_CATEGORIES = [
    "Incomplete Information",
    "Irrelevant Information",
    "Erroneous Information",
    "Incomplete or Missing Response",
    "Inaccurate or Misunderstood Response",
    "Irrelevant or Off-Topic Response",
    "Overly Verbose Response"
]

VALID_TAG2_CATEGORIES = [
    "Insufficient or Incomplete Information Retrieval",
    "Data Insufficiency in Retrieval",
    "Relevance Gaps in Retrieval",
    "Irrelevant Information Retrieval",
    "Erroneous Information Retrieval",
    "Omission of Key Information",
    "Lack of Specificity",
    "Specificity and Precision Errors",
    "Partial Coverage and Temporal Issues",
    "Lack of Practicality",
    "Contextual Understanding Errors",
    "Factual Inaccuracies",
    "Incorrect and Incomplete Answers",
    "Misinterpretation of Queries and Information",
    "Entity and Concept Confusion",
    "Irrelevant Content and Topic Drift",
    "Off-Topic and Redundant Responses",
    "Content and Context Misalignment",
    "Overly Complex and Redundant Response"
]

# 计算每个样本的准确率和 F1 分数的函数
def calculate_metrics(golden, output):
    metrics = {}

    for tag in ['tag1', 'tag2']:
        golden_tags = set(golden[tag])
        output_tags = set(output[tag])
        
        # 根据不同tag选择对应的有效类别
        valid_categories = VALID_TAG1_CATEGORIES if tag == 'tag1' else VALID_TAG2_CATEGORIES
        
        # 只保留有效类别的标签
        golden_tags = {tag for tag in golden_tags if tag in valid_categories}
        output_tags = {tag for tag in output_tags if tag in valid_categories}
        
        # 计算匹配标签
        
        matches = golden_tags.intersection(output_tags)
        accuracy = len(matches) / len(golden_tags) if golden_tags else 0
        
        # 计算 TP、FP、FN 以获得 F1 分数
        TP = len(matches)  # 真阳性
        FP = len(output_tags - golden_tags)  # 假阳性
        FN = len(golden_tags - output_tags)  # 假阴性
        
        # 计算精确率和召回率
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # 计算 F1 分数
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        metrics[tag] = {
            "accuracy": accuracy,
            "f1": f1
        }
        
    return metrics

# 计算 ROUGE 分数
def calculate_text_metrics(golden_text, output_text):
    try:
        rouge = Rouge()
        # 限制文本长度，防止递归过深
        max_length = 1000  # 设置一个合理的最大长度
        golden_text = golden_text[:max_length] if len(golden_text) > max_length else golden_text
        output_text = output_text[:max_length] if len(output_text) > max_length else output_text
        
        rouge_scores = rouge.get_scores(output_text, golden_text)[0]
        return {
            "rouge": {
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"]
            }
        }
    except Exception as e:
        print(f"Warning: Error calculating ROUGE scores: {str(e)}")
        return {
            "rouge": {
                "rouge-1": 0,
                "rouge-2": 0,
                "rouge-l": 0
            }
        }

# 添加F1计算函数的定义
def calculate_f1(correct, total, predicted_total):
    if total == 0 or predicted_total == 0:
        return 0
    precision = correct / predicted_total if predicted_total > 0 else 0
    recall = correct / total if total > 0 else 0
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# 添加ROUGE计算函数
def get_rouge_scores(golden_text, output_text):
    try:
        rouge = Rouge()
        # 同样限制文本长度
        max_length = 1000
        golden_text = golden_text[:max_length] if len(golden_text) > max_length else golden_text
        output_text = output_text[:max_length] if len(output_text) > max_length else output_text
        
        scores = rouge.get_scores(output_text, golden_text)[0]
        return {
            "rouge-1": round(scores["rouge-1"]["f"], 4),
            "rouge-2": round(scores["rouge-2"]["f"], 4),
            "rouge-l": round(scores["rouge-l"]["f"], 4)
        }
    except Exception as e:
        print(f"Warning: Error calculating ROUGE scores: {str(e)}")
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0
        }

file_path = 'your_path_to_generated_result/predict_qwen-2.5-7b-instruct_predict_1900_130.json'  # change to the path of your predict JSON

# 初始化总体指标统计
total_stats = {
    'tag1': {
        'correct': 0,
        'total': 0,
        'predicted_total': 0
    },
    'tag2': {
        'correct': 0,
        'total': 0,
        'predicted_total': 0
    }
}

# 初始化类别统计
category_stats = {
    'tag1': {category: {'correct': 0, 'total': 0, 'predicted_total': 0} 
            for category in VALID_TAG1_CATEGORIES},
    'tag2': {category: {'correct': 0, 'total': 0, 'predicted_total': 0} 
            for category in VALID_TAG2_CATEGORIES}
}

# 初始化ROUGE分数累计
rouge_scores_sum = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
valid_rouge_count = 0

# 初始化Judgement统计
judgement_stats = {
    'overall': {'correct': 0, 'total': 0},
    'tag1': {'correct': 0, 'total': 0},
    'tag2': {'correct': 0, 'total': 0},
    'correct_case': {'correct': 0, 'total': 0}  # 新增Correct判断统计
}

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)
            golden = eval(data['golden'])
            output = data['output']
            
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            # 更新整体Judgement统计
            judgement_stats['overall']['total'] += 1
            if golden['Judgement'] == output['Judgement']:
                judgement_stats['overall']['correct'] += 1

            # 统计Correct判断的准确率
            if golden['Judgement'] == 'Correct':
                judgement_stats['correct_case']['total'] += 1
                if output['Judgement'] == 'Correct':
                    judgement_stats['correct_case']['correct'] += 1

            # 更新tag1和tag2的Judgement统计
            # 只有当golden中有对应的tag时才统计
            if 'tag1' in golden:
                judgement_stats['tag1']['total'] += 1
                if golden['Judgement'] == 'Error':  # 如果golden判断为Error
                    if output['Judgement'] == 'Error' and 'tag1' in output:  # 且output也判断为Error且包含tag1
                        judgement_stats['tag1']['correct'] += 1

            if 'tag2' in golden:
                judgement_stats['tag2']['total'] += 1
                if golden['Judgement'] == 'Error':  # 如果golden判断为Error
                    if output['Judgement'] == 'Error' and 'tag2' in output:  # 且output也判断为Error且包含tag2
                        judgement_stats['tag2']['correct'] += 1

            if golden['Judgement'] != output['Judgement']:
                # 如果 Judgement 不一致，准确率和 F1 设为 0
                metrics = {tag: {"accuracy": 0, "f1": 0} for tag in ['tag1', 'tag2']}
            elif golden['Judgement'] == 'Correct':
                # 如果 Judgement 均为 Correct，跳过该样本
                continue
            elif golden['Judgement'] == 'Error' and output['Judgement'] == 'Error':
                # 如果均为 Error，则计算指标
                metrics = calculate_metrics(golden, output)
                # 计算ROUGE分数并累计
                rouge_scores = get_rouge_scores(str(golden), str(output))
                rouge_scores_sum["rouge-1"] += rouge_scores["rouge-1"]
                rouge_scores_sum["rouge-2"] += rouge_scores["rouge-2"]
                rouge_scores_sum["rouge-l"] += rouge_scores["rouge-l"]
                valid_rouge_count += 1
            else:
                # 如果没有被处理，安全起见设为 0
                metrics = {tag: {"accuracy": 0, "f1": 0} for tag in ['tag1', 'tag2']}

            # 更新总体统计
            for tag in ['tag1', 'tag2']:
                valid_categories = VALID_TAG1_CATEGORIES if tag == 'tag1' else VALID_TAG2_CATEGORIES
                golden_tags = set(t for t in golden.get(tag, []) if t in valid_categories)
                output_tags = set(t for t in output.get(tag, []) if t in valid_categories)
                
                matches = golden_tags.intersection(output_tags)
                
                total_stats[tag]['total'] += len(golden_tags)
                total_stats[tag]['predicted_total'] += len(output_tags)
                total_stats[tag]['correct'] += len(matches)

                # 更新类别统计
                for category in valid_categories:
                    if category in golden_tags:
                        category_stats[tag][category]['total'] += 1
                    if category in output_tags:
                        category_stats[tag][category]['predicted_total'] += 1
                    if category in matches:
                        category_stats[tag][category]['correct'] += 1

        except Exception as e:
            print(f"Error processing line: {str(e)}")
            parse_failed_count += 1
            continue

# 计算平均ROUGE分数
average_rouge = {
    "rouge-1": rouge_scores_sum["rouge-1"] / valid_rouge_count if valid_rouge_count > 0 else 0,
    "rouge-2": rouge_scores_sum["rouge-2"] / valid_rouge_count if valid_rouge_count > 0 else 0,
    "rouge-l": rouge_scores_sum["rouge-l"] / valid_rouge_count if valid_rouge_count > 0 else 0
}

# 计算总体指标
overall_metrics = {
    tag: {
        "accuracy": round(total_stats[tag]['correct'] / total_stats[tag]['total'], 4) 
                   if total_stats[tag]['total'] > 0 else 0,
        "f1": round(calculate_f1(total_stats[tag]['correct'], 
                               total_stats[tag]['total'], 
                               total_stats[tag]['predicted_total']), 4),
        "rouge": {
            "rouge-1": round(average_rouge["rouge-1"], 4),
            "rouge-2": round(average_rouge["rouge-2"], 4),
            "rouge-l": round(average_rouge["rouge-l"], 4)
        }
    }
    for tag in ['tag1', 'tag2']
}

# 计算类别指标
category_metrics = {
    tag: {
        category: {
            "accuracy": round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0,
            "f1": round(calculate_f1(stats['correct'], stats['total'], stats['predicted_total']), 4),
            "rouge": {
                "rouge-1": round(average_rouge["rouge-1"], 4),
                "rouge-2": round(average_rouge["rouge-2"], 4),
                "rouge-l": round(average_rouge["rouge-l"], 4)
            }
        }
        for category, stats in category_stats[tag].items()
    }
    for tag in ['tag1', 'tag2']
}

# 计算Judgement准确率
judgement_accuracy = {
    'overall': round(judgement_stats['overall']['correct'] / judgement_stats['overall']['total'], 4) if judgement_stats['overall']['total'] > 0 else 0,
    'tag1': round(judgement_stats['tag1']['correct'] / judgement_stats['tag1']['total'], 4) if judgement_stats['tag1']['total'] > 0 else 0,
    'tag2': round(judgement_stats['tag2']['correct'] / judgement_stats['tag2']['total'], 4) if judgement_stats['tag2']['total'] > 0 else 0,
    'correct_case': round(judgement_stats['correct_case']['correct'] / judgement_stats['correct_case']['total'], 4) if judgement_stats['correct_case']['total'] > 0 else 0
}

# 在整合结果之前，计算总体的accuracy、f1和rouge
overall_accuracy = round((overall_metrics["tag1"]["accuracy"] + overall_metrics["tag2"]["accuracy"]) / 2, 4)
overall_f1 = round((overall_metrics["tag1"]["f1"] + overall_metrics["tag2"]["f1"]) / 2, 4)
overall_rouge = {
    "rouge-1": round(average_rouge["rouge-1"], 4),
    "rouge-2": round(average_rouge["rouge-2"], 4),
    "rouge-l": round(average_rouge["rouge-l"], 4)
}

# 修改final_results的构建
final_results = {
    "overall": {
        "accuracy": overall_accuracy,  # 添加总体accuracy
        "f1": overall_f1,             # 添加总体f1
        "rouge": overall_rouge,       # 添加总体rouge
        "judgement_accuracy": judgement_accuracy['overall'],
        "correct_judgement_accuracy": judgement_accuracy['correct_case'],
        "tag1": {
            **overall_metrics["tag1"],
            "judgement_accuracy": judgement_accuracy['tag1']
        },
        "tag2": {
            **overall_metrics["tag2"],
            "judgement_accuracy": judgement_accuracy['tag2']
        }
       
    },
    "category_metrics": {
        "tag1": category_metrics["tag1"],
        "tag2": category_metrics["tag2"]
    }
}

# 从输入文件路径中提取基础文件名
input_filename = file_path.split('/')[-1]
base_filename = input_filename.replace('.json', '')

# 构建输出文件路径
output_path = f'/your output path/{base_filename}_metric_result.json'

# 将结果保存到JSON文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

# 在控制台输出详细的评估结果
print("=================== 总体指标 ===================")
print(f"\n整体指标:")
print(f"准确率: {overall_accuracy:.4f}")
print(f"F1分数: {overall_f1:.4f}")
print("ROUGE分数:")
print(f"  ROUGE-1: {overall_rouge['rouge-1']:.4f}")
print(f"  ROUGE-2: {overall_rouge['rouge-2']:.4f}")
print(f"  ROUGE-L: {overall_rouge['rouge-l']:.4f}")
print(f"总体Judgement准确率: {judgement_accuracy['overall']:.4f}")

print("\nTag1 总体指标:")
print(f"Judgement准确率: {judgement_accuracy['tag1']:.4f}")
print(f"准确率: {overall_metrics['tag1']['accuracy']:.4f}")
print(f"F1分数: {overall_metrics['tag1']['f1']:.4f}")
print("ROUGE分数:")
print(f"  ROUGE-1: {overall_metrics['tag1']['rouge']['rouge-1']:.4f}")
print(f"  ROUGE-2: {overall_metrics['tag1']['rouge']['rouge-2']:.4f}")
print(f"  ROUGE-L: {overall_metrics['tag1']['rouge']['rouge-l']:.4f}")

print("\nTag2 总体指标:")
print(f"Judgement准确率: {judgement_accuracy['tag2']:.4f}")
print(f"准确率: {overall_metrics['tag2']['accuracy']:.4f}")
print(f"F1分数: {overall_metrics['tag2']['f1']:.4f}")
print("ROUGE分数:")
print(f"  ROUGE-1: {overall_metrics['tag2']['rouge']['rouge-1']:.4f}")
print(f"  ROUGE-2: {overall_metrics['tag2']['rouge']['rouge-2']:.4f}")
print(f"  ROUGE-L: {overall_metrics['tag2']['rouge']['rouge-l']:.4f}")

print("\n=================== 类别详细指标 ===================")
print("\nTag1 类别指标:")
for category, metrics in category_metrics["tag1"].items():
    print(f"\n{category}:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  ROUGE分数:")
    print(f"    ROUGE-1: {metrics['rouge']['rouge-1']:.4f}")
    print(f"    ROUGE-2: {metrics['rouge']['rouge-2']:.4f}")
    print(f"    ROUGE-L: {metrics['rouge']['rouge-l']:.4f}")

print("\nTag2 类别指标:")
for category, metrics in category_metrics["tag2"].items():
    print(f"\n{category}:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  ROUGE分数:")
    print(f"    ROUGE-1: {metrics['rouge']['rouge-1']:.4f}")
    print(f"    ROUGE-2: {metrics['rouge']['rouge-2']:.4f}")
    print(f"    ROUGE-L: {metrics['rouge']['rouge-l']:.4f}")

# 输出总的准确率和 F1 分数
print(f"\nJudgement 准确率: {judgement_accuracy['overall']:.2f}%")
print(f"解析失败的样本数量: {parse_failed_count}")
