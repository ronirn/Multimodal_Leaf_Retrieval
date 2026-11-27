// LeafMatch AI - small UI behaviours.

document.addEventListener("DOMContentLoaded", () => {
  // Medicinal Plant Collection: arrow button scrolls by one card; wraps to start.
  const track = document.getElementById("collection-track");
  const next = document.getElementById("collection-next");
  if (track && next) {
    next.addEventListener("click", () => {
      const card = track.querySelector(".plant-card");
      const step = card ? card.getBoundingClientRect().width + 20 : 220;
      const atEnd = track.scrollLeft + track.clientWidth >= track.scrollWidth - 4;
      track.scrollTo({ left: atEnd ? 0 : track.scrollLeft + step, behavior: "smooth" });
    });
  }
});
