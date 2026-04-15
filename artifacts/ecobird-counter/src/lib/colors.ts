export const SPECIES_COLORS: Record<string, string> = {
  Sparrow: "#3B82F6",
  Finch: "#22C55E",
  Warbler: "#EF4444",
  Robin: "#EAB308",
  Kingfisher: "#06B6D4",
  Swallow: "#8B5CF6",
  Eagle: "#F97316",
  Unknown: "#9CA3AF"
};

export function getSpeciesColor(species: string): string {
  return SPECIES_COLORS[species] || SPECIES_COLORS.Unknown;
}
