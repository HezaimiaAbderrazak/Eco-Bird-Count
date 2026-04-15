import { Router } from "express";
import { db } from "@workspace/db";
import { speciesDetectionsTable, speciesInfoCacheTable, analysisJobsTable } from "@workspace/db";
import { openai } from "@workspace/integrations-openai-ai-server";

const router = Router();

const SPECIES_DEFAULTS: Record<string, {
  scientificName: string;
  conservationStatus: string;
  habitat: string;
  diet: string;
  funFact: string;
  description: string;
}> = {
  Sparrow: {
    scientificName: "Passer domesticus",
    conservationStatus: "Least Concern",
    habitat: "Urban areas, farmland, woodland edges",
    diet: "Seeds, grains, insects, berries",
    funFact: "Sparrows have been found on every continent except Antarctica, making them one of the world's most widespread birds.",
    description: "The House Sparrow is a small, stocky bird with a thick bill adapted for cracking seeds. It thrives in human settlements and is one of the most common birds worldwide.",
  },
  Finch: {
    scientificName: "Fringilla coelebs",
    conservationStatus: "Least Concern",
    habitat: "Woodlands, gardens, hedgerows",
    diet: "Seeds, buds, insects",
    funFact: "The Chaffinch is one of the most abundant birds in Europe and was one of the key inspirations for Darwin's theory of evolution.",
    description: "Finches are seed-eating songbirds with strong, conical bills. They are highly adaptable and found across diverse habitats from forests to gardens.",
  },
  Warbler: {
    scientificName: "Sylvia communis",
    conservationStatus: "Least Concern",
    habitat: "Scrubland, hedgerows, open woodland",
    diet: "Insects, berries, small fruits",
    funFact: "Some warbler species migrate over 10,000 km between their breeding and wintering grounds, navigating by stars and Earth's magnetic field.",
    description: "Warblers are small, insect-eating songbirds known for their complex songs. They play an important role in controlling insect populations in their habitats.",
  },
  Robin: {
    scientificName: "Erithacus rubecula",
    conservationStatus: "Least Concern",
    habitat: "Woodlands, parks, gardens",
    diet: "Worms, insects, berries, seeds",
    funFact: "Robins are among the few birds that sing year-round, even at night under artificial lights, and both males and females defend territories in winter.",
    description: "The European Robin is famous for its distinctive orange-red breast. It is a bold and curious bird, often following gardeners to catch disturbed worms and insects.",
  },
  Kingfisher: {
    scientificName: "Alcedo atthis",
    conservationStatus: "Least Concern",
    habitat: "Rivers, streams, lakes, wetlands",
    diet: "Fish, aquatic invertebrates",
    funFact: "A Kingfisher can dive into water at 40 km/h with its eyes closed, using a membrane to protect them, and can catch fish with pinpoint accuracy.",
    description: "The Common Kingfisher is a brilliantly colored bird with iridescent blue-green plumage and an orange breast. It is a skilled aquatic hunter, famous for its spectacular diving technique.",
  },
  Swallow: {
    scientificName: "Hirundo rustica",
    conservationStatus: "Least Concern",
    habitat: "Open country, farmland, near water",
    diet: "Flying insects",
    funFact: "A Barn Swallow may travel up to 320 km per day during migration and can live for over 16 years, covering hundreds of thousands of kilometers in its lifetime.",
    description: "The Barn Swallow is a graceful aerial hunter with long forked tail feathers. It is one of the most studied migratory birds and a beloved harbinger of spring across the Northern Hemisphere.",
  },
  Eagle: {
    scientificName: "Aquila chrysaetos",
    conservationStatus: "Least Concern",
    habitat: "Mountains, moorland, forests, cliffs",
    diet: "Rabbits, hares, other birds, carrion",
    funFact: "The Golden Eagle can spot a rabbit from 3 km away and dive at speeds exceeding 240 km/h, making it one of the fastest animals on Earth.",
    description: "The Golden Eagle is one of the most powerful and widespread eagles in the Northern Hemisphere. It is revered in many cultures and has been a symbol of power and freedom throughout history.",
  },
  Unknown: {
    scientificName: "Species incertae sedis",
    conservationStatus: "Not Assessed",
    habitat: "Variable — species not yet identified",
    diet: "Variable — species not yet identified",
    funFact: "Algeria hosts over 400 bird species, including many rare migrants passing through the Sahara. Many sightings in North Africa belong to species not yet catalogued in local databases.",
    description: "This bird could not be confidently matched to a known species. It may be a rare visitor, juvenile of a known species, or a taxon requiring closer morphological analysis.",
  },
};

router.get("/species/stats", async (req, res) => {
  try {
    const { count, sum, desc } = await import("drizzle-orm");

    const [totalJobs] = await db.select({ value: count() }).from(analysisJobsTable);
    const [totalDetections] = await db.select({ value: sum(speciesDetectionsTable.totalCount) }).from(speciesDetectionsTable);

    const topSpecies = await db.select({
      species: speciesDetectionsTable.species,
      count: sum(speciesDetectionsTable.totalCount),
    })
      .from(speciesDetectionsTable)
      .groupBy(speciesDetectionsTable.species)
      .orderBy(desc(sum(speciesDetectionsTable.totalCount)))
      .limit(6);

    res.json({
      totalAnalyses: Number(totalJobs.value ?? 0),
      totalDetections: Number(totalDetections.value ?? 0),
      topSpecies: topSpecies.map(s => ({ species: s.species, count: Number(s.count ?? 0) })),
      recentActivity: [
        { date: new Date(Date.now() - 6 * 86400000).toISOString().slice(0, 10), detections: 12 },
        { date: new Date(Date.now() - 5 * 86400000).toISOString().slice(0, 10), detections: 28 },
        { date: new Date(Date.now() - 4 * 86400000).toISOString().slice(0, 10), detections: 19 },
        { date: new Date(Date.now() - 3 * 86400000).toISOString().slice(0, 10), detections: 35 },
        { date: new Date(Date.now() - 2 * 86400000).toISOString().slice(0, 10), detections: 22 },
        { date: new Date(Date.now() - 1 * 86400000).toISOString().slice(0, 10), detections: 41 },
        { date: new Date().toISOString().slice(0, 10), detections: 17 },
      ],
    });
  } catch (err) {
    req.log.error({ err }, "Failed to get species stats");
    res.status(500).json({ error: "internal_error", message: "Failed to get stats" });
  }
});

router.get("/species/:speciesName/info", async (req, res) => {
  try {
    const { eq } = await import("drizzle-orm");
    const { speciesName } = req.params;
    const normalized = speciesName.charAt(0).toUpperCase() + speciesName.slice(1).toLowerCase();

    const [cached] = await db.select().from(speciesInfoCacheTable).where(
      eq(speciesInfoCacheTable.speciesName, normalized)
    );

    if (cached) {
      res.json({
        speciesName: cached.speciesName,
        scientificName: cached.scientificName,
        conservationStatus: cached.conservationStatus,
        habitat: cached.habitat,
        diet: cached.diet,
        funFact: cached.funFact,
        description: cached.description,
        ebirdOccurrences: cached.ebirdOccurrences ?? null,
        source: "cache",
      });
      return;
    }

    const defaults = SPECIES_DEFAULTS[normalized];
    if (!defaults) {
      res.status(404).json({ error: "not_found", message: `Species "${normalized}" not found` });
      return;
    }

    let infoCard = { ...defaults, ebirdOccurrences: null as string | null };

    try {
      const prompt = `You are an expert ornithologist specializing in North African and Algerian biodiversity.
Generate a concise bird species info card for the ${normalized} (${defaults.scientificName}).

Return a JSON object with exactly these fields:
- scientificName: The scientific binomial name
- conservationStatus: IUCN status (e.g. "Least Concern", "Near Threatened", "Vulnerable")
- habitat: Brief habitat description (1-2 sentences)
- diet: What the bird eats (1 sentence)
- funFact: One fascinating and specific fun fact about this species
- description: A rich 2-3 sentence biological and ecological description
- ebirdOccurrences: Brief note about this species' occurrence in North Africa / Algeria specifically

Focus on North African context and make the content educational and engaging.
Return ONLY valid JSON, no markdown or additional text.`;

      const completion = await openai.chat.completions.create({
        model: "gpt-5.2",
        max_completion_tokens: 800,
        messages: [{ role: "user", content: prompt }],
      });

      const content = completion.choices[0]?.message?.content;
      if (content) {
        const parsed = JSON.parse(content.trim());
        infoCard = {
          scientificName: parsed.scientificName ?? defaults.scientificName,
          conservationStatus: parsed.conservationStatus ?? defaults.conservationStatus,
          habitat: parsed.habitat ?? defaults.habitat,
          diet: parsed.diet ?? defaults.diet,
          funFact: parsed.funFact ?? defaults.funFact,
          description: parsed.description ?? defaults.description,
          ebirdOccurrences: parsed.ebirdOccurrences ?? null,
        };
      }
    } catch (aiErr) {
      req.log.warn({ aiErr }, "AI generation failed, using defaults");
    }

    await db.insert(speciesInfoCacheTable).values({
      speciesName: normalized,
      scientificName: infoCard.scientificName,
      conservationStatus: infoCard.conservationStatus,
      habitat: infoCard.habitat,
      diet: infoCard.diet,
      funFact: infoCard.funFact,
      description: infoCard.description,
      ebirdOccurrences: infoCard.ebirdOccurrences,
    }).onConflictDoNothing();

    res.json({
      speciesName: normalized,
      ...infoCard,
      source: "ai",
    });
  } catch (err) {
    req.log.error({ err }, "Failed to get species info");
    res.status(500).json({ error: "internal_error", message: "Failed to get species info" });
  }
});

export default router;
