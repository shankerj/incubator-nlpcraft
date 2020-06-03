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

package org.apache.nlpcraft.server.nlp.enrichers.ml

import io.opencensus.trace.Span
import org.apache.nlpcraft.common.NCService
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.nlp.{NCNlpSentence, NCNlpSentenceNote}
import org.apache.nlpcraft.server.ml.NCMlManager
import org.apache.nlpcraft.server.nlp.enrichers.NCServerEnricher

object NCMlEnricher extends NCServerEnricher {
    @volatile private var url: String = _

    private object Config extends NCConfigurable {
        lazy val url: String = getStringOrElse("nlpcraft.server.ml.url", "http://localhost:5000")
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { span ⇒
        url = Config.url

        if (url.last == '/')
            url = url.dropRight(1)

        addTags(span, "url" → url)

        // Tries to access spaCy proxy server.
        // TODO: add health check.

        logger.info(s"ML server connected: $url")

        super.start()
    }

    override def stop(parent: Span): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    private def substitute(words: Seq[String], idx: Int, repl: String): String =
        words.zipWithIndex.map { case (w, i) ⇒ if (idx == i) repl else w }.mkString(" ")

    private def mark(ns: NCNlpSentence, idx: Int, elemId: String, valueOpt: Option[String]): Unit = {
        val tok = ns(idx)

        val note =
            valueOpt match {
                case Some(value) ⇒ NCNlpSentenceNote(Seq(tok.index), elemId, "value" → value)
                case None ⇒ NCNlpSentenceNote(Seq(tok.index), elemId)
            }

        tok.add(note)
    }

//    private def intersect(suggs: Seq[NCMlSynonymInfoMdo], syns: Seq[NCProbableSynonym]): Option[String] =
//        suggs.flatMap(
//            sug ⇒ syns.flatMap(syn ⇒ if (syn.word == sug.word) syn.value else None).toStream.headOption
//        ).toStream.headOption


    override def enrich(ns: NCNlpSentence, parent: Span): Unit = {
        ns.mlCfg match {
            case Some(cfg) ⇒
                // TODO: other names.
                val nn = ns.filter(_.pos.startsWith("N"))

                if (nn.nonEmpty) {
                    val normTxt = ns.map(_.origText).mkString(" ")

                    nn.foreach(n ⇒ {
                        val nIdx = ns.indexOf(n)

                        val suggs = NCMlManager.suggest(normTxt, nIdx, 0.5, 10) // TODO:

                        logger.info(s"Suggestions for main sentence [text=$normTxt, nn=${n.origText}, suggestions=${suggs.mkString(",")}]")

//                        cfg.synonyms.foreach { case (elemId, syns) ⇒
//                            var found = false
//
//                            for ((synStem, info) ← syns if !found)
//                                if (synStem == n.stem || suggs.exists(_.stem == n.stem)) {
//                                    mark(ns, nIdx, elemId, info.value)
//
//                                    found = true
//                                }
//
//                            if (!found) {
//                                val elemExamples = cfg.examples(elemId)
//
//                                case class Holder(stem: String, score: Double, value: Option[String])
//
//                                val allExData =
//                                    elemExamples.map { case (exampleToks, substIdx) ⇒
//                                        val subs = substitute(exampleToks, substIdx, n.origText)
//                                        val suggs = NCMlManager.suggest(subs, substIdx, 0.5, 10)
//
//                                        syns.
//                                            filter { case (synStem, _) ⇒ suggs.exists(_.stem == synStem) }.
//                                            map { case (synStem, info) ⇒
//                                                Holder(stem = synStem, score = info.score, value = info.value)
//                                            }.toSeq
//                                    }.toSeq.flatten.groupBy(p ⇒ p)
//
//                                allExData.find { case (_, hs) ⇒ hs.size == elemExamples.size } match {
//                                    case Some((h, _)) ⇒ mark(ns, n)
//                                    case None ⇒ // No-op.
//                                }
//                            }
//
//
//                            syns.foreach { case (stem, info) ⇒
//                            }
//                        }
//
//                        cfg.mlElements.flatMap { case (elemId, syns) ⇒
//                            intersect(sugg, syns) match {
//                                case Some(v) ⇒ Some(elemId, v)
//                                case None ⇒ None
//                            }
//                        }.toStream.headOption match {
//                            case Some((elemId, v)) ⇒
//                                val tok = ns(idx)
//
//                                // TODO: value,
//                                tok.add(NCNlpSentenceNote(Seq(tok.index), elemId))
//
//
//                            case None ⇒
//                                cfg.examples.foreach { case (elemId, elemExamples) ⇒
//                                    println("elemId="+elemId)
//                                    println("elemExamples="+elemExamples)
//                                    val all =
//                                        elemExamples.forall { case (example, idx) ⇒
//                                            val subs = substitute(example, idx, n.origText)
//                                            val suggs = NCMlManager.ask(subs, idx, 0.5, 10)
//
//                                            // TODO: value
//                                            val ok = intersect(suggs, cfg.mlElements(elemId)).getOrElse(false)
//
//                                            logger.info(s"Suggestions for examples [subs=$subs, i=$idx, nn=${n.origText}, suggestions=${suggs.mkString(",")}, ok=$ok]")
//
//                                            ok
//                                        }
//
//                                    if (all) {
//                                        mark(ns, idx, elemId)
//                                    }
//                                }
//                        }
                    })
                }
            case None ⇒ // No-op.
        }
    }

}
