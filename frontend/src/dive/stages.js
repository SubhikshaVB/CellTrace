// Single source of truth for the Dive: our project flow, in order.
export const STAGES = [
  { id: "s00", num: "00", name: "SPECIMEN", kicker: "STAGE 00 · THE DIVE BEGINS" },
  { id: "s01", num: "01", name: "VALIDATE", kicker: "STAGE 01 · TRUST, THEN COMPUTE" },
  { id: "s02", num: "02", name: "EXTRACT", kicker: "STAGE 02 · CUT OUT EVERY SUSPECT" },
  { id: "s03", num: "03", name: "STANDARDIZE", kicker: "STAGE 03 · DEVELOP THE EVIDENCE" },
  { id: "s04", num: "04", name: "EMBED", kicker: "STAGE 04 · EVERY CELL GETS A FACE" },
  { id: "s05", num: "05", name: "CONNECT", kicker: "STAGE 05 · NO CELL IS AN ISLAND" },
  { id: "s06", num: "06", name: "TRAIN", kicker: "STAGE 06 · TEACHING THE MACHINE" },
  { id: "s07", num: "07", name: "TRACK", kicker: "STAGE 07 · THE VERDICT" },
  { id: "s08", num: "08", name: "EXPORT", kicker: "STAGE 08 · BAG THE EVIDENCE" },
];

export const STAGE_IDS = STAGES.map((s) => s.id);
