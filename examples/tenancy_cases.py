"""Example Swiss tenancy law cases for testing.

Each case is a realistic scenario based on common Mietrecht disputes.
Sources: OpenCaseLaw search on "Mietrecht Kündigung" themes.
"""

EXAMPLES = [
    {
        "name": "Termination Without Official Form",
        "claimant_role": "tenant",
        "claim_type": "termination_validity",
        "canton": "CH",
        "language": "de",
        "raw_text": """Ich bin Mieter einer 3.5-Zimmer-Wohnung in Zürich seit dem 1. April 2019. 
Am 15. Januar 2024 habe ich ein Kündigungsschreiben von meinem Vermieter erhalten. 
Das Schreiben war ein einfacher Brief ohne das offizielle kantonale Formular. 
Der Vermieter begründet die Kündigung mit "Eigenbedarf für seinen Sohn". 
Ich habe keine weiteren Informationen erhalten. Ich bezahle monatlich CHF 1'850 Miete.
Die Kündigung soll per 31. März 2024 wirksam werden.
Ich frage mich, ob diese Kündigung gültig ist, da das offizielle Formular fehlt.""",
        "attached_documents": ["Kündigungsschreiben vom 15.01.2024 (einfacher Brief ohne Formular)"],
    },
    {
        "name": "Rent Increase After Renovation",
        "claimant_role": "tenant",
        "claim_type": "rent_increase",
        "canton": "CH",
        "language": "de",
        "raw_text": """Ich wohne seit 2020 in einer 4-Zimmer-Wohnung in Bern. Der Vermieter hat im Sommer 2023 
die Küche renoviert (neue Schränke, neuer Boden). Im September 2023 erhielt ich eine 
Mietzinserhöhung von CHF 2'200 auf CHF 2'650 pro Monat, gültig ab 1. Januar 2024.
Das offizielle Formular für die Mietzinserhöhung wurde verwendet, aber die Erhöhung 
scheint mir überhöht. Die Renovation der Küche hat laut Vermieter CHF 25'000 gekostet.
Ich habe die Erhöhung nicht angefochten, weil ich unsicher war. Jetzt frage ich mich,
ob die Erhöhung rechtmässig ist und ob ich noch etwas dagegen tun kann.""",
        "attached_documents": ["Mietzinserhöhungsformular vom 15.09.2023"],
    },
    {
        "name": "Defective Heating in Winter",
        "claimant_role": "tenant",
        "claim_type": "defect_remediation",
        "canton": "CH",
        "language": "de",
        "raw_text": """Ich bin Mieter einer Wohnung in Basel seit 2021. Seit November 2023 funktioniert 
die Heizung nur unregelmässig. Die Raumtemperatur fällt oft unter 18°C.
Ich habe den Vermieter mehrfach schriftlich (per E-Mail am 20.11.2023, 5.12.2023, 
und 2.1.2024) über den Mangel informiert. Der Vermieter hat einmal einen Techniker 
geschickt (am 10.12.2023), aber das Problem wurde nicht behoben.
Ich habe die Miete im Januar 2024 um 20% gekürzt (von CHF 1'500 auf CHF 1'200).
Der Vermieter droht nun mit Kündigung wegen Zahlungsverzug.
Ich möchte wissen, ob meine Mietzinshinterlegung rechtmässig war und ob die 
Kündigungsdrohung zulässig ist.""",
        "attached_documents": ["E-Mail-Korrespondenz (3 Nachrichten)", "Technikerbericht vom 10.12.2023"],
    },
    {
        "name": "Retaliatory Termination After Complaint",
        "claimant_role": "tenant",
        "claim_type": "termination_validity",
        "canton": "CH",
        "language": "de",
        "raw_text": """Ich habe seit 2018 eine Wohnung in Genf gemietet. Am 5. Februar 2024 habe ich beim 
Vermieter schriftlich die Reparatur eines Wasserschadens im Badezimmer gefordert.
Am 20. Februar 2024 erhielt ich eine Kündigung per 31. Mai 2024, mit der 
Begründung "Umbauarbeiten". Ich glaube, die Kündigung ist eine Vergeltung 
für meine Mängelrüge (Rachekündigung).
Ich bezahle CHF 2'100 pro Monat. Ich habe immer pünktlich bezahlt.
Die Mängelrüge und die Kündigung liegen nur 15 Tage auseinander.""",
        "attached_documents": ["Mängelrüge vom 05.02.2024", "Kündigung vom 20.02.2024"],
    },
    {
        "name": "Deposit Not Returned",
        "claimant_role": "tenant",
        "claim_type": "deposit_dispute",
        "canton": "CH",
        "language": "de",
        "raw_text": """Ich bin am 31. Dezember 2023 aus meiner Wohnung in Luzern ausgezogen, nachdem ich 
dort 4 Jahre gewohnt habe. Ich habe eine Mietkaution von CHF 5'400 (3 Monatsmieten) 
auf einem Mietkautionskonto hinterlegt.
Bei der Wohnungsübergabe am 28.12.2023 wurde ein Abnahmeprotokoll erstellt, das 
nur minimale Gebrauchsspuren dokumentiert (keine Schäden).
Jetzt, 6 Monate nach Auszug, hat die Bank die Kaution immer noch nicht freigegeben.
Der Vermieter behauptet, es gäbe "versteckte Schäden" und verweigert die Zustimmung
zur Freigabe. Ich habe keine Fotos vom Auszugszustand.
Ich möchte wissen, wie ich meine Kaution zurückbekomme und ob das Verhalten des 
Vermieters rechtmässig ist.""",
        "attached_documents": ["Abnahmeprotokoll vom 28.12.2023"],
    },
]
