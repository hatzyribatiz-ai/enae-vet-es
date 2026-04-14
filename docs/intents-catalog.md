# Intents Catalog — Veterinary Clinic Chatbot

> 20 intents aligned with the 10 acceptance conversations.
> Each intent includes: name, description, example utterance, and expected behaviour.

---

## Intent list

| # | Intent | Description | Example utterance | Expected behaviour |
|---|--------|-------------|-------------------|--------------------|
| 1 | **Salutation** | User greets the bot or asks what it can do. | "Hi, what can you help me with?" | Welcome message; explain scope (sterilisation/castration booking, pre-op info). Clarify: no emergencies, no routine consults. |
| 2 | **OutOfScopeGeneralConsult** | User asks for diagnosis, prescription, or routine illness consult. | "My cat has a cough for three days — can you prescribe something?" | Politely refuse clinical advice; redirect to a vet or emergency service if severe. Stay within assistant scope. |
| 3 | **QueryDropOffWindow** | User asks what time to bring the pet on surgery day. | "What time should I bring my cat on surgery day?" | Cats: 08:00–09:00. Dogs: 09:00–10:30. Mention carrier requirement for cats. |
| 4 | **QueryPickUpTime** | User asks when to pick up the pet after surgery. | "When can I pick up my dog after castration?" | Dogs: ~12:00. Cats: ~15:00. Note: mention flexibility if times are inconvenient. |
| 5 | **QueryPreOpInstructions** | User asks about fasting, preparation, or what to bring. | "How long should my dog fast before surgery?" | Food: 8–12 h before. Water: until 1–2 h before. Bring signed consent + documentation. |
| 6 | **QueryMedicalRequirements** | User asks about blood test or medical prerequisites. | "My dog is 8 years old. Is a blood test required?" | Mandatory for >6 years. Recommended for younger. Clinic can refer to partner lab. |
| 7 | **QueryEligibility** | User asks if their pet is eligible for surgery under specific conditions (age, breed, etc.). | "What if she were 5?" | Distinguish mandatory vs recommended based on age threshold; consistent with spec. |
| 8 | **RequestEmergencyCare** | User reports an emergency (bleeding, accident, poisoning). | "My dog was hit by a car and is bleeding. Can you book me for tomorrow?" | Do NOT book. Direct to emergency vet immediately. Be empathetic and brief. |
| 9 | **BookProcedure** | User wants to schedule a sterilisation or castration. | "I want to book a spay for my female dog." | Collect species, preferred day, any concerns. Check eligibility rules (heat, age). Proceed to availability if applicable. |
| 10 | **CheckAvailability** | User asks about available days/capacity for surgery. | "I need to spay my cat next week. What days do you have capacity?" | Invoke check_availability tool. Return specific available days with constraints. |
| 11 | **QueryHeatRestriction** | User mentions their female dog is in heat and wants to book. | "I want to book a spay for my female dog. She is currently in heat." | Reject: cannot spay during/around heat. Must wait ~2 months after heat ends. Cats: no restriction. |
| 12 | **QueryCapacity** | User asks about capacity when the day is already busy. | "Can we do my large dog's surgery this Thursday if you already have two other dogs that day?" | Use tool; if blocked by 2-dog limit or minute budget, explain clearly and suggest alternatives. |
| 13 | **QueryWeekendAvailability** | User asks if surgery is available on weekends. | "Are you open on weekends for surgery?" | Clarify: surgery Mon–Thu only. The clinic may have other hours but surgery is weekday only. |
| 14 | **HumanHandoff** | User explicitly asks to speak with a person. | "I'd rather speak to a person about my invoice." | Acknowledge; provide escalation path (phone/WhatsApp during opening hours). Collect callback preference if wanted. |
| 15 | **QueryPostOpCare** | User asks about post-operative care instructions. | "What should I do after my cat's surgery?" | Quiet/warm environment, water after 4–5 h, soft food after 6–8 h, no products on wound except chlorhexidine, internal stitches (no removal). |
| 16 | **QueryTransportRequirements** | User asks how to transport the pet to the clinic. | "Do I need a carrier for my cat?" | Cats: rigid carrier required (no cardboard/fabric), towel inside, one per carrier. Dogs: collar/harness + leash, muzzle if aggressive. |
| 17 | **QueryCost** | User asks about prices or payment methods. | "How much does a cat spay cost?" | Provide general info if available; mention payment by cash or card. Microchip/vaccine are extra costs. |
| 18 | **QueryDocumentation** | User asks what documents to bring on surgery day. | "What paperwork do I need for the surgery?" | Signed informed consent, pet passport or health card. Microchip + rabies vaccine mandatory by law. |
| 19 | **CancelOrReschedule** | User wants to cancel or change an appointment. | "I need to cancel my appointment for Thursday." | Remind: 24 h notice required to avoid surcharge. Offer to reschedule. |
| 20 | **QueryMedication** | User asks about post-surgery medication. | "Will my dog need medication after the operation?" | Dogs: capsule/liquid ~6 h after, anti-inflammatory 24 h after. Cats: syrup/tablet 24 h after if tolerated. Do NOT prescribe specific doses. |

---

## Conversation → Intent mapping

| Conv. | Theme | Primary intents |
|-------|-------|-----------------|
| 1 | Greeting & scope | Salutation, OutOfScopeGeneralConsult |
| 2 | Drop-off windows + memory | QueryDropOffWindow |
| 3 | Blood test / age | QueryMedicalRequirements, QueryEligibility |
| 4 | Emergency / out of scope | RequestEmergencyCare |
| 5 | Heat restriction | BookProcedure, QueryHeatRestriction |
| 6 | Pick-up times + memory | QueryPickUpTime |
| 7 | Human handoff | HumanHandoff |
| 8 | Availability (tool) | CheckAvailability, QueryWeekendAvailability |
| 9 | Capacity / Tetris (tool) | QueryCapacity, CheckAvailability, BookProcedure |
| 10 | Pre-op / fasting (RAG) | QueryPreOpInstructions |

---

## Notes

- Intents can co-occur in a single conversation (e.g., BookProcedure + CheckAvailability).
- The system prompt and/or RAG cover the domain knowledge needed for all intents.
- Memory (session_id) is required for conversations 2 and 6 where the user changes species mid-conversation.
- The check_availability tool is invoked for intents CheckAvailability and QueryCapacity (conversations 8 and 9).
