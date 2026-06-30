function render(nodes, edges, containerId, drawEdges=false) {

  const width = 1000, height = 600;
  const legendWidth = 180
  const margin = { top: 20, right: 20, bottom: 40, left: 50 };
  const innerWidth = width - margin.left - margin.right - legendWidth;
  const innerHeight = height - margin.top - margin.bottom;
  

  const svg = d3.select(containerId)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);

  const xScale = d3.scaleLinear()
    .domain(d3.extent(nodes, d => d.x))
    .range([0, innerWidth]);

  const yScale = d3.scaleLinear()
    .domain(d3.extent(nodes, d => d.y))
    .range([innerHeight, 0]);

  const labels = Array.from(new Set(nodes.map(d => d.label)));
  
  const legendLabels = [...new Set(nodes.map(d => d.label))]
  .sort((a, b) => a - b);

  const colorScale = d3.scaleOrdinal()
    .domain(labels)
    .range(d3.schemeSet3);

  const nodeMap = new Map(nodes.map(d => [d.id, d]));

  if (drawEdges) {
    g.selectAll("line")
      .data(edges)
      .enter()
      .append("line")
      .attr("x1", d => xScale(nodeMap.get(d.source).x))
      .attr("y1", d => yScale(nodeMap.get(d.source).y))
      .attr("x2", d => xScale(nodeMap.get(d.target).x))
      .attr("y2", d => yScale(nodeMap.get(d.target).y))
      .attr("stroke", "#333")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4 2");
  }

  g.selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("cx", d => xScale(d.x))
    .attr("cy", d => yScale(d.y))
    .attr("r", 3)
    .attr("fill", d => d.pred_correct === false
        ? "red"
        : colorScale(d.label)
    )
    .attr("stroke", "#aaa")
    .attr("stroke-width", 0.5);

  const crosses = nodes.filter(d => d.symbol === "cross");

  g.selectAll(".cross")
    .data(crosses)
    .enter()
    .append("path")
    .attr("d", d3.symbol().type(d3.symbolCross).size(80))
    .attr("fill", d =>
        d.pred_correct === false
        ? "red"
        : colorScale(d.label)
    )
    .attr("stroke", "black")        
    .attr("stroke-width", 1.0)
    .attr("transform", d =>
        `translate(${xScale(d.x)}, ${yScale(d.y)})`
    );

    const legend = svg.append("g")
    .attr("transform", `translate(${innerWidth + margin.left + 70}, 20)`);

    const legendItem = legend.selectAll(".legend-item")
    .data(legendLabels)
    .enter()
    .append("g")
    .attr("class", "legend-item")
    .attr("transform", (d, i) => `translate(0, ${i * 20})`);

    legendItem.append("circle")
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("r", 5)
    .attr("fill", d => colorScale(d))
    .attr("stroke", "#aaa")
    .attr("stroke-width", 0.5);

    // label text
    legendItem.append("text")
    .attr("x", 10)
    .attr("y", 4)
    .text(d => d)
    .style("font-family", "sans-serif")
    .style("font-size", "12px");

}
