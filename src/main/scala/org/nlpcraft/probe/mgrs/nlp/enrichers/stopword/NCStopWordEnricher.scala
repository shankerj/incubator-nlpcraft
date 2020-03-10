/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.nlpcraft.probe.mgrs.nlp.enrichers.stopword

import java.io.Serializable

import io.opencensus.trace.Span
import org.nlpcraft.common.nlp.core.NCNlpCoreManager
import org.nlpcraft.common.nlp.{NCNlpSentence, NCNlpSentenceToken}
import org.nlpcraft.common.{NCE, NCService, U}
import org.nlpcraft.probe.mgrs.NCModelDecorator
import org.nlpcraft.probe.mgrs.nlp.NCProbeEnricher

import scala.annotation.tailrec
import scala.collection.{Map, Seq}

/**
  * Stop words enricher.
  */
object NCStopWordEnricher extends NCProbeEnricher {
    private final val POSES: Seq[String] = Seq("DT", "PRP", "PRP$", "WDT", "WP", "WP$", "WRB", "TO", "IN")

    private final val GEO_TYPES = Set(
        "nlpcraft:continent",
        "nlpcraft:subcontinent",
        "nlpcraft:country",
        "nlpcraft:metro",
        "nlpcraft:region",
        "nlpcraft:city"
    )

    @throws[NCE]
    private final val GEO_PRE_WORDS: Seq[Seq[String]] =
    // NOTE: stemmatisation is done already by generator.
        U.readTextResource(s"context/geo_pre_words.txt", "UTF-8", logger).
            toSeq.map(_.split(" ").toSeq).sortBy(-_.size)

    private final val GEO_KIND_STOPS =
        Map(
            "nlpcraft:city" → Seq("city", "town"),
            "nlpcraft:country" → Seq("country", "land", "countryside", "area", "territory"),
            "nlpcraft:region" → Seq("region", "area", "state", "county", "district", "ground", "territory"),
            "nlpcraft:continent" → Seq("continent", "land", "area")
        ).map(p ⇒ p._1 → p._2.map(NCNlpCoreManager.stem))

    private final val NUM_PREFIX_STOPS = Seq(
        "is",
        "was",
        "were",
        "are",
        "with value",
        "might be",
        "would be",
        "has value",
        "can be",
        "should be",
        "must be"
    ).map(NCNlpCoreManager.stem)

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        super.start()
    }
    
    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    /**
      * Marks tokens as stopwords before token with given type if these tokens are suitable by given predicate.
      *
      * @param sen Sentence.
      * @param noteType Note type.
      */
    private def markBefore(sen: NCNlpSentence, noteType: String): Boolean = {
        var res = false

        for (note ← sen.getNotes(noteType) if note.tokenFrom > 0) {
            val part = sen.
                take(note.tokenFrom).
                reverse.
                takeWhile(t ⇒ t.isStopWord || t.isNlp && POSES.contains(t.pos)).
                filter(!_.isStopWord)

            if (part.nonEmpty) {
                part.foreach(_.addStopReason(note))

                res = true
            }
        }

        res
    }

    /**
      * Processes geo tokens. Sets additional stopwords.
      *
      * @param ns Sentence.
      */
    private def processGeo(ns: NCNlpSentence): Boolean = {
        var res = false

        // 1. Marks some specific words before GEO (like 'origin for London')
        for (note ← GEO_TYPES.flatMap(ns.getNotes)) {
            val toks = ns.
                take(note.tokenFrom).
                reverse.
                takeWhile(t ⇒ t.isStopWord || t.isNlp).
                reverse

            if (toks.nonEmpty) {
                val stems = toks.map(_.stem)

                GEO_PRE_WORDS.find(stems.endsWith) match {
                    case Some(words) ⇒
                        val part = toks.reverse.take(words.size).filter(!_.isStopWord)

                        if (part.nonEmpty) {
                            res = true

                            part.foreach(_.addStopReason(note))
                        }
                    case None ⇒ // No-op.
                }
            }
        }

        // 2. Marks some specific words before and after GEO (like 'city London')
        GEO_KIND_STOPS.foreach { case (typ, stops) ⇒
            for (geoNote ← ns.getNotes(typ)) {
                def process(toks: Seq[NCNlpSentenceToken]): Unit =
                    toks.find(!_.isStopWord) match {
                        case Some(t) ⇒ if (stops.contains(t.stem)) {
                            res = true

                            t.addStopReason(geoNote)
                        }
                        case None ⇒ // No-op.
                    }

                process(ns.filter(_.index > geoNote.tokenTo))
                process(ns.filter(_.index < geoNote.tokenFrom).reverse)
            }
        }

        // 3. Marks stop-words like prepositions before.
        GEO_TYPES.foreach(t ⇒ {
            if (markBefore(ns, t))
                res = true
        })

        res
    }

    /**
      * Processes nums tokens. Sets additional stopwords.
      *
      * @param ns Sentence.
      */
    private def processNums(ns: NCNlpSentence): Boolean = {
        var res = false

        // Try to find words from configured list before numeric condition and mark them as STOP words.
        ns.getNotes("nlpcraft:num").foreach(numNote ⇒ {
            val before = ns.filter(_.index < numNote.tokenFrom)

            before.indices.map(i ⇒ before.drop(i)).find(
                seq ⇒ seq.forall(
                    t ⇒
                        t.isStopWord ||
                        (!t.isBracketed && !t.isQuoted)) &&
                    NUM_PREFIX_STOPS.contains(seq.filter(!_.isStopWord).map(_.stem).mkString(" "))
            ) match {
                case Some(seq) ⇒
                    val toks = seq.filter(!_.isStopWord)

                    if (toks.nonEmpty) {
                        res = true

                        toks.foreach(_.addStopReason(numNote))
                    }
                case None ⇒ // No-op.
            }
        })

        res
    }

    /**
      * Processes dates. Sets additional stopwords.
      *
      * @param ns Sentence.
      */
    private def processDate(ns: NCNlpSentence): Boolean = markBefore(ns, "nlpcraft:date")

    /**
      * Marks as stopwords, words with POS from configured list, which also placed before another stop words.
      */
    private def processCommonStops(mdl: NCModelDecorator, ns: NCNlpSentence): Boolean = {
        var res = false

        /**
          * Marks as stopwords, words with POS from configured list, which also placed before another stop words.
          */
        @tailrec
        def processCommonStops0(mdl: NCModelDecorator, ns: NCNlpSentence): Boolean = {
            val max = ns.size - 1
            var stop = true

            for (
                (tok, idx) ← ns.zipWithIndex
                if idx != max &&
                    !tok.isStopWord &&
                    !mdl.excludedStopWordsStems.contains(tok.stem) &&
                    POSES.contains(tok.pos) &&
                    ns(idx + 1).isStopWord
            ) {
                tok.markAsStop()

                res = true

                stop = false
            }

            if (stop) true else processCommonStops0(mdl, ns)
        }

        processCommonStops0(mdl, ns)

        res
    }

    @throws[NCE]
    override def enrich(mdl: NCModelDecorator, ns: NCNlpSentence, senMeta: Map[String, Serializable], parent: Span = null): Boolean = {
        def mark(stems: Set[String], f: Boolean): Boolean = {
            val part = ns.filter(t ⇒ stems.contains(t.stem))

            if (part.nonEmpty) {
                part.foreach(_.getNlpNote += "stopWord" → f)

                true
            }
            else
                false
        }
    
        startScopedSpan("enrich", parent, "srvReqId" → ns.srvReqId, "modelId" → mdl.model.getId, "txt" → ns.text) { _ ⇒
            Seq(
                mark(mdl.excludedStopWordsStems, f = false),
                mark(mdl.additionalStopWordsStems, f = true),
                processGeo(ns),
                processDate(ns),
                processNums(ns),
                processCommonStops(mdl, ns)
            ).contains(true)
        }
    }
}
