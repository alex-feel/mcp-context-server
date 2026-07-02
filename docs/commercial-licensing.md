# Commercial Licensing

MCP Context Server is distributed under the [Elastic License 2.0](../LICENSE) (ELv2). This page explains, in plain language, what the license lets you do for free, what requires a commercial agreement, and how to obtain one. The [license text](../LICENSE) is the authoritative source; this page is interpretive guidance from the licensor.

## The Model in One Paragraph

The software is free for everyone to use, in any setup, at any scale, including inside commercial companies and as part of paid work. The single reserved right is offering the software itself to third parties as a hosted or managed service: providing users with access to any substantial set of its features or functionality as a service requires a commercial license from the licensor. Releases up to and including v2.2.2 were published under the MIT License and remain available under it; ELv2 applies from v3.0.0 onward.

## Permitted Without Any Agreement (Free)

- Running the server for yourself, on any machine, for any purpose, including commercial ones.
- Deploying the server inside your company as memory/context infrastructure for your own agents, products, and employees — regardless of company size or revenue.
- A consultant or contractor installing and configuring the server for a client, where the client uses it internally.
- Modifying the source, building derivative works, and redistributing copies or forks, as long as the license terms accompany them and notices stay intact.
- Embedding the server as an internal component of your own product or service, where your users interact with YOUR product's functionality and never receive access to a substantial set of THIS software's own features (its MCP tools for storing, searching, and retrieving context).

## Requires a Commercial Agreement

- Offering MCP Context Server (or a modified version of it) to third parties as a hosted or managed service — for example, a cloud "memory for agents" API, a multi-tenant context-storage service, or a hosted MCP endpoint whose users store and retrieve their own context through this software's functionality.
- Any offering where what your customers effectively buy is access to this software's feature set, whether exposed directly or through a thin wrapper.

## Boundary Cases

- **Operating a client's instance on your infrastructure (managed service providers).** Elastic's own guidance treats this as dependent on what the customer receives: when your customers get access to a substantial set of the software's functionality, the limitation applies. As the licensor's interpretation: operating a dedicated, single-tenant instance on behalf of one client, solely for that client's internal use (the outsourced-operations equivalent of the permitted contractor case), is tolerated; anything multi-tenant, or marketed as a context-storage/memory service, requires an agreement. When in doubt, ask — see below.
- **Building a product on top.** Using the server as invisible backend infrastructure of a larger product is permitted (see above). The line is crossed when the value your users receive from your offering is, in substance, this software's own store/search/retrieve functionality.

## How to Obtain a Commercial License

Email [alexfeel@protonmail.com](mailto:alexfeel@protonmail.com) with a short description of the intended offering (what your users would receive, single- or multi-tenant, expected scale). Terms are negotiated individually.

## Contributions

External contributions are accepted only under a contributor license agreement that grants the licensor the right to license the contribution under any terms, including commercial ones — see [CONTRIBUTING.md](../CONTRIBUTING.md). This is what keeps the dual-licensing model legally sound.

## Third-Party Code

The vendored TurboQuant compression component (`app/compression/providers/turboquant/`) includes code under the MIT License; its notice is preserved in [THIRD_PARTY_LICENSES.md](../app/compression/providers/turboquant/THIRD_PARTY_LICENSES.md) and continues to apply to that portion.
