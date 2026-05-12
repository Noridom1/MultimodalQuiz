/** Checkbox-driven type selection helpers. */

export const QUIZ_QUESTION_TYPE_LABELS = {
  multiple_choice: "Multiple choice",
  true_false: "True/False",
  fill_in_blank: "Fill in the blank",
  matching: "Matching",
};

const TYPE_ORDER = ["multiple_choice", "true_false", "fill_in_blank", "matching"];

export function normalizeSelectedQuestionTypes(selectedTypes) {
  const list = Array.isArray(selectedTypes) ? selectedTypes : [];
  return TYPE_ORDER.filter((id) => list.includes(id));
}

export function buildEqualDistribution(selectedTypes) {
  const normalized = normalizeSelectedQuestionTypes(selectedTypes);
  if (normalized.length === 0) {
    return { multiple_choice: 1 };
  }
  const weight = Number((1 / normalized.length).toFixed(6));
  const distribution = {};
  normalized.forEach((id, index) => {
    distribution[id] = index === normalized.length - 1
      ? Number((1 - weight * (normalized.length - 1)).toFixed(6))
      : weight;
  });
  return distribution;
}
