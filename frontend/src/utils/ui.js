export function heroStyle(item, index) {
  const gradients = [
    "linear-gradient(135deg, rgba(90,103,255,.8), rgba(12,18,44,.95))",
    "linear-gradient(135deg, rgba(223,120,62,.85), rgba(41,24,14,.92))",
    "linear-gradient(135deg, rgba(78,179,148,.85), rgba(11,35,34,.95))",
  ];
  return item?.hero_image
    ? { backgroundImage: `linear-gradient(180deg, rgba(19,23,31,.05), rgba(19,23,31,.9)), url(${item.hero_image})` }
    : { backgroundImage: gradients[index % gradients.length] };
}

export function formatBytes(value) {
  if (!value) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
