
# Given a different format ground-truth and model-output, uniform it
def clean_gt_dependencies(deps):
    transformed = []
    for item in deps:
        combined_dict = {}
        for sub_item in item:
            combined_dict.update(sub_item)
        transformed.append(combined_dict)
    return transformed
